# simulate_dwts_voting_swap.py

import pandas as pd
import numpy as np
import re

FAN_ESTIMATES_CSV = "dwts_fan_estimates.csv"


def rank_series_best1_is_rank1(s):
    """Return rank series where larger values get smaller rank number (1 = best).
       Ties given the minimum rank (method='min') to be consistent.
    """
    return s.rank(ascending=False, method='min')

def simulate_rank_based_week(df_week, n_eliminate=1):
    """Season 1-2 style (rank-based) for a single week subset df_week:
       - Rank by judge percent and fan_mean (both higher = better)
       - Combined rank = 0.5 * (judge_rank + fan_rank)
       - Eliminate the n_eliminate contestants with the worst combined_rank (i.e., largest)
       Tie-break deterministic: lower fan_mean worse, then lower judge percent.
       Returns list of eliminated celebrity_name(s).
    """
    df = df_week.copy()
    # Check required columns
    if df['fan_mean'].isnull().any() or df['percent_judge_score'].isnull().any():
        # if missing data for some entries, remove them (cannot predict)
        df = df.dropna(subset=['fan_mean', 'percent_judge_score'])
    if df.shape[0] == 0:
        return []

    df['rank_judge'] = rank_series_best1_is_rank1(df['percent_judge_score'])
    df['rank_fan'] = rank_series_best1_is_rank1(df['fan_mean'])
    df['combined_rank'] = 0.5 * (df['rank_judge'] + df['rank_fan'])

    # Sort by combined_rank descending (worst combined_rank first),
    # tie-break with fan_mean ascending (lower fan_mean worse), then judge percent ascending.
    df = df.sort_values(by=['combined_rank', 'fan_mean', 'percent_judge_score'],
                        ascending=[False, True, True])
    eliminated = df.head(n_eliminate)['celebrity_name'].tolist()
    return eliminated

def simulate_percent_based_week(df_week, n_eliminate=1):
    """Season 3+ style (percentage-based) for a single week subset df_week:
       - combined_percent = 0.5 * (percent_judge_score + fan_mean)
       - Eliminate n_eliminate contestants with lowest combined_percent
       Tie-break deterministic: lower fan_mean then lower judge percent.
       Returns list of eliminated celebrity_name(s).
    """
    df = df_week.copy()
    if df['fan_mean'].isnull().any() or df['percent_judge_score'].isnull().any():
        df = df.dropna(subset=['fan_mean', 'percent_judge_score'])
    if df.shape[0] == 0:
        return []

    df['combined_percent'] = 0.5 * (df['percent_judge_score'] + df['fan_mean'])
    df = df.sort_values(by=['combined_percent', 'fan_mean', 'percent_judge_score'],
                        ascending=[True, True, True])
    eliminated = df.head(n_eliminate)['celebrity_name'].tolist()
    return eliminated

def simulate_bottom_two_judges_week(df_week, n_eliminate=1):
    """Season 28-34 style:
       - Identify bottom two by combined_percent = 0.5*(percent_judge_score + fan_mean)
       - During live show judges vote between those two to select whom to eliminate.
       - If explicit judge-vote columns exist in df_week, they are used to decide.
       - Otherwise, deterministic tie-break: contestant with lower percent_judge_score is eliminated.
       Returns list of eliminated celebrity_name(s).
    """
    df = df_week.copy()
    if df['fan_mean'].isnull().any() or df['percent_judge_score'].isnull().any():
        df = df.dropna(subset=['fan_mean', 'percent_judge_score'])
    if df.shape[0] == 0:
        return []

    df['combined_percent'] = 0.5 * (df['percent_judge_score'] + df['fan_mean'])
    df = df.sort_values(by=['combined_percent', 'fan_mean', 'percent_judge_score'],
                        ascending=[True, True, True])
    bottom = df.head(2).copy()
    if bottom.shape[0] == 0:
        return []
    if bottom.shape[0] == 1:
        return [bottom.iloc[0]['celebrity_name']]

    # Try to detect judge-vote columns (heuristic)
    vote_cols = [c for c in df.columns if re.search(r'judge.*vote|vote.*judge', c, flags=re.IGNORECASE)]
    if 'judge_vote' in df.columns:
        vote_cols.append('judge_vote')
    vote_cols = list(dict.fromkeys(vote_cols))

    # If vote columns found, aggregate votes for the two and pick the one with more elimination votes
    if vote_cols:
        vote_counts = {}
        for _, row in bottom.iterrows():
            name = row['celebrity_name']
            total_votes = 0
            for vc in vote_cols:
                v = row.get(vc)
                # Accept numeric votes; if boolean or 0/1, coerce to int
                try:
                    total_votes += int(v)
                except Exception:
                    # If column stores judge name or similar, check equality by name
                    # e.g., a column may contain the eliminated celebrity name for that judge
                    if isinstance(v, str) and v.strip() == name:
                        total_votes += 1
            vote_counts[name] = total_votes

        # select elimination(s) by highest vote count (higher = eliminated)
        max_votes = max(vote_counts.values())
        selected = [n for n, ct in vote_counts.items() if ct == max_votes]
        # If exactly n_eliminate chosen, return them; otherwise fall through to deterministic rule
        if len(selected) == n_eliminate:
            return selected
        if len(selected) > n_eliminate:
            # If multiple tied with top votes, break by judge percent (lower percent worse)
            tied_df = bottom[bottom['celebrity_name'].isin(selected)]
            tied_df = tied_df.sort_values(by='percent_judge_score', ascending=True)
            return tied_df.head(n_eliminate)['celebrity_name'].tolist()

    # Fallback: eliminate bottom-two member(s) with lower percent_judge_score
    bottom = bottom.sort_values(by='percent_judge_score', ascending=True)
    return bottom.head(n_eliminate)['celebrity_name'].tolist()

def get_metadata_for_names(df_week, names, cols):
    """Return dict mapping col -> list of values (matching order of names).
       Missing values yield None.
    """
    res = {c: [] for c in cols}
    for n in names:
        subset = df_week[df_week['celebrity_name'] == n]
        if subset.empty:
            for c in cols:
                res[c].append(None)
        else:
            row = subset.iloc[0]
            for c in cols:
                res[c].append(row.get(c, None))
    return res

def run_simulation_swap(fan_df, seasons_to_swap_rank_to_percent, seasons_to_swap_percent_to_rank, debug=False):
    results = []

    # Group by season and week (sort by week)
    grouped = fan_df.groupby(['season', 'week'])
    for (season, week), df_week in grouped:
        df_week = df_week.copy()
        # Determine how many actual eliminations occurred that week (could be 0, 1, 2, etc.)
        n_actual_elim = int(df_week['eliminated'].sum())
        if n_actual_elim == 0:
            # Nothing to compare (no elimination), skip
            continue

        if season in seasons_to_swap_percent_to_rank:
            predicted = simulate_rank_based_week(df_week, n_eliminate=n_actual_elim)
            predicted_method = 'rank_based'
        elif season in seasons_to_swap_rank_to_percent:
            predicted = simulate_percent_based_week(df_week, n_eliminate=n_actual_elim)
            predicted_method = 'percent_based'
        else:
            continue

        actual_elimed = df_week.loc[df_week['eliminated']==1, 'celebrity_name'].tolist()
        actual_set = set(actual_elimed)
        predicted_set = set(predicted)

        match = (actual_set == predicted_set)

        # collect metadata for actual eliminated contestants
        meta = get_metadata_for_names(df_week, actual_elimed,
                                      ['ballroom_partner', 'celebrity_industry', 'celebrity_age_during_season'])

        results.append({
            'season': season,
            'week': week,
            'n_actual_elim': n_actual_elim,
            'actual_eliminated': actual_elimed,
            'predicted_eliminated': predicted,
            'pred_method': predicted_method,
            'match': match,
            'n_predicted': len(predicted),
            'actual_ballroom_partner': meta['ballroom_partner'],
            'actual_celebrity_industry': meta['celebrity_industry'],
            'actual_celebrity_age_during_season': meta['celebrity_age_during_season']
        })

        if debug:
            print(f"Season {season} Week {week} | actual: {actual_elimed} | predicted({predicted_method}): {predicted} | match={match}")

    results_df = pd.DataFrame(results)
    return results_df

def run_simulation_compare_methods_for_seasons(fan_df, seasons, debug=False):
    """For the given seasons, run three methods (rank-based, percent-based, bottom-two+jg vote)
       for each week and compare predictions to actual eliminated set.
       Returns a DataFrame with an entry per week per method.
    """
    rows = []
    grouped = fan_df.groupby(['season', 'week'])
    for (season, week), df_week in grouped:
        if season not in seasons:
            continue
        df_week = df_week.copy()
        n_actual_elim = int(df_week['eliminated'].sum())
        if n_actual_elim == 0:
            continue

        actual_elimed = df_week.loc[df_week['eliminated'] == 1, 'celebrity_name'].tolist()
        actual_set = set(actual_elimed)

        # collect metadata once per week
        actual_meta = get_metadata_for_names(df_week, actual_elimed,
                                             ['ballroom_partner', 'celebrity_industry', 'celebrity_age_during_season'])

        # rank-based
        pred_rank = simulate_rank_based_week(df_week, n_eliminate=n_actual_elim)
        # percent-based
        pred_percent = simulate_percent_based_week(df_week, n_eliminate=n_actual_elim)
        # bottom-two + judges
        pred_bottom_judges = simulate_bottom_two_judges_week(df_week, n_eliminate=n_actual_elim)

        for method_name, pred in [('rank_based', pred_rank),
                                  ('percent_based', pred_percent),
                                  ('bottom_two_judges', pred_bottom_judges)]:
            pred_set = set(pred)
            rows.append({
                'season': season,
                'week': week,
                'method': method_name,
                'n_actual_elim': n_actual_elim,
                'actual_eliminated': actual_elimed,
                'predicted_eliminated': pred,
                'match': (pred_set == actual_set),
                'n_predicted': len(pred),
                'actual_ballroom_partner': actual_meta['ballroom_partner'],
                'actual_celebrity_industry': actual_meta['celebrity_industry'],
                'actual_celebrity_age_during_season': actual_meta['celebrity_age_during_season']
            })

        if debug:
            print(f"Season {season} Week {week} | actual: {actual_elimed} | rank: {pred_rank} | percent: {pred_percent} | bottom+judges: {pred_bottom_judges}")

    res_df = pd.DataFrame(rows)
    return res_df

def main():
    df_fans = pd.read_csv(FAN_ESTIMATES_CSV)

    # Ensure column names are as expected; drop rows lacking needed fields
    required = ['season','week','celebrity_name','percent_judge_score','fan_mean','eliminated',
                'ballroom_partner','celebrity_industry','celebrity_age_during_season']
    for r in required:
        if r not in df_fans.columns:
            raise RuntimeError(f"Missing required column in fan estimates file: {r}")

    # Convert to numeric
    df_fans['season'] = df_fans['season'].astype(int)
    df_fans['week'] = df_fans['week'].astype(int)
    df_fans['percent_judge_score'] = pd.to_numeric(df_fans['percent_judge_score'], errors='coerce')
    df_fans['fan_mean'] = pd.to_numeric(df_fans['fan_mean'], errors='coerce')
    # keep age numeric when possible
    df_fans['celebrity_age_during_season'] = pd.to_numeric(df_fans['celebrity_age_during_season'], errors='coerce')
    df_fans['eliminated'] = pd.to_numeric(df_fans['eliminated'], errors='coerce').fillna(0).astype(int)

    # Define seasons to swap:
    seasons_1_2 = [1, 2]
    seasons_3_27 = list(range(3, 28))

    # Simulate seasons 3-27 with season 1-2 system, and seasons 1-2 with season 3+ system
    results_df = run_simulation_swap(df_fans, seasons_to_swap_rank_to_percent=seasons_1_2,
                                     seasons_to_swap_percent_to_rank=seasons_3_27,
                                     debug=False)

    # Summaries
    print("Total weeks evaluated:", len(results_df))
    matches = results_df['match'].sum()
    print(f"Weeks where predicted elimination set exactly matched actual elimination set: {matches} / {len(results_df)}")
    print(f"Match rate: {matches / len(results_df):.3%}")

    # Show mismatches
    mismatches = results_df[~results_df['match']].copy()
    print(f"Number of mismatches: {len(mismatches)}")
    if len(mismatches) > 0:
        print("\nExamples of mismatches (first 20):")
        for idx, row in mismatches.head(23).iterrows():
            print(f"Season {row['season']} Week {row['week']}: actual {row['actual_eliminated']} vs predicted {row['predicted_eliminated']} ({row['pred_method']})")

    # Results to CSV 
    results_df.to_csv("simulation_swap_results.csv", index=False)
    print("\nResults written to simulation_swap_results.csv")

    # New: compare methods on seasons 28-34
    seasons_28_34 = list(range(28, 35))
    comp28_df = run_simulation_compare_methods_for_seasons(df_fans, seasons_28_34, debug=False)
    if len(comp28_df) == 0:
        print("\nNo elimination weeks found for seasons 28-34 in the input data.")
    else:
        print("\nSeason 28-34 method comparison:")
        summary = comp28_df.groupby('method')['match'].agg(['sum','count'])
        summary['match_rate'] = summary['sum'] / summary['count']
        print(summary)
        comp28_df.to_csv("simulation_28_34_comparison.csv", index=False)
        print("Per-week, per-method results written to simulation_28_34_comparison.csv")

if __name__ == "__main__":
    main()