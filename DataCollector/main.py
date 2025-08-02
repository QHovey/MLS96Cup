import pandas as pd
from itscalledsoccer.client import AmericanSoccerAnalysis
from collections import defaultdict
import sys
import itertools
import os

def calculate_h2h_points(team_list, league, season, conference_name, all_teams, asa_client):
    """
    Finds all matches between a given list of teams, calculates a
    head-to-head table, and returns the top team and the generated dataframes.

    This function now separates the logic for exporting all scheduled matches
    from the logic for calculating standings based on played matches.

    Args:
        team_list (list): A list of team names for the analysis.
        league (str): The league to search within.
        season (str): The season to search within.
        conference_name (str): The name of the conference for display.
        all_teams (pd.DataFrame): DataFrame containing all team data.
        asa_client (AmericanSoccerAnalysis): The API client instance.

    Returns:
        tuple: (top_team_name, standings_df, all_conference_matches_df)
               The third element is a DataFrame of ALL matches for CSV export.
    """
    print(f"\n{'='*20} {conference_name.upper()} CONFERENCE ANALYSIS ({season}) {'='*20}")
    if len(team_list) < 2:
        print("Error: Please provide at least two teams in the list.")
        return None, pd.DataFrame(), pd.DataFrame()

    try:
        # 1. Find the team IDs for all teams in the input list
        team_ids = {}
        valid_team_list = []
        for team_name in team_list:
            try:
                team_id = all_teams[all_teams['team_name'] == team_name]['team_id'].iloc[0]
                team_ids[team_name] = team_id
                valid_team_list.append(team_name)
            except IndexError:
                print(f"\nWarning: Could not find team '{team_name}'. It will be skipped.")
        
        if len(team_ids) < 2:
            print("Error: Not enough valid teams found to perform an analysis.")
            return None, pd.DataFrame(), pd.DataFrame()
            
        print(f"Found IDs for {len(team_ids)} teams.")
        team_id_list = list(team_ids.values())

        # 2. Get all games for the specified season, filtering for Regular Season only
        print(f"Fetching Regular Season game data for the {season} {league.upper()} season...")
        # Fetch completed games (defaults to status='FullTime')
        completed_games = pd.DataFrame(asa_client.get_games(leagues=[league], seasons=[season], stages=['Regular Season']))
        
        # Fetch upcoming games
        print(f"Fetching upcoming (PreMatch) game data for the {season} {league.upper()} season...")
        upcoming_games = pd.DataFrame(asa_client.get_games(leagues=[league], seasons=[season], stages=['Regular Season'], status=['PreMatch']))

        # Combine the two dataframes
        all_games = pd.concat([completed_games, upcoming_games], ignore_index=True)


        if all_games.empty:
            print(f"No game data found for the {season} {league.upper()} season.")
            return None, pd.DataFrame(), pd.DataFrame()

        # 3. Filter for ALL matches between conference teams for the export file.
        # This includes games that are not yet played ('PreMatch').
        all_conference_matches = all_games[
            (all_games['home_team_id'].isin(team_id_list)) & 
            (all_games['away_team_id'].isin(team_id_list))
        ].copy()

        # Add team names for readability to the export DataFrame
        all_conference_matches = pd.merge(all_conference_matches, all_teams[['team_id', 'team_name']], left_on='home_team_id', right_on='team_id', how='left').rename(columns={'team_name': 'home_team'}).drop('team_id', axis=1)
        all_conference_matches = pd.merge(all_conference_matches, all_teams[['team_id', 'team_name']], left_on='away_team_id', right_on='team_id', how='left').rename(columns={'team_name': 'away_team'}).drop('team_id', axis=1)
        
        # Convert score columns to nullable integers to handle NaNs and prevent floats in CSV
        if 'home_score' in all_conference_matches.columns:
            all_conference_matches['home_score'] = all_conference_matches['home_score'].astype(pd.Int64Dtype())
        if 'away_score' in all_conference_matches.columns:
            all_conference_matches['away_score'] = all_conference_matches['away_score'].astype(pd.Int64Dtype())

        # 4. Filter for PLAYED games to be used in standings calculation.
        # A played game is one that has a non-null score.
        played_games_df = all_conference_matches.dropna(subset=['home_score', 'away_score']).copy()
        
        if played_games_df.empty:
            print(f"\nNo played matches found between the specified teams in the {season} {league.upper()} regular season to calculate standings.")
            # Create an empty standings dataframe but return all scheduled matches for export
            empty_standings = pd.DataFrame(columns=['team_name', 'games_played', 'wins', 'losses', 'draws', 'points', 'goals_scored', 'goals_conceded', 'goal_differential'])
            return None, empty_standings, all_conference_matches

        # Convert scores to integers for calculation
        played_games_df['home_score'] = played_games_df['home_score'].astype(int)
        played_games_df['away_score'] = played_games_df['away_score'].astype(int)

        # 5. Advanced Match Filtering Logic for STANDINGS CALCULATION (from played games)
        print("Filtering for the most recent home and away match for each team pairing (for standings)...")
        games_to_keep = []
        for team1_name, team2_name in itertools.combinations(valid_team_list, 2):
            team1_id = team_ids[team1_name]
            team2_id = team_ids[team2_name]

            # Find most recent game where team1 is home vs team2
            t1_home_games = played_games_df[(played_games_df['home_team_id'] == team1_id) & (played_games_df['away_team_id'] == team2_id)]
            if not t1_home_games.empty:
                games_to_keep.append(t1_home_games.sort_values(by='date_time_utc', ascending=False).head(1))

            # Find most recent game where team2 is home vs team1
            t2_home_games = played_games_df[(played_games_df['home_team_id'] == team2_id) & (played_games_df['away_team_id'] == team1_id)]
            if not t2_home_games.empty:
                games_to_keep.append(t2_home_games.sort_values(by='date_time_utc', ascending=False).head(1))

        if not games_to_keep:
            # This case should be rare given the check for played_games_df above, but it's good practice.
            print(f"\nNo valid matches found after applying home/away filtering rules.")
            empty_standings = pd.DataFrame(columns=['team_name', 'games_played', 'wins', 'losses', 'draws', 'points', 'goals_scored', 'goals_conceded', 'goal_differential'])
            return None, empty_standings, all_conference_matches

        matches_df_for_calc = pd.concat(games_to_keep).drop_duplicates().reset_index(drop=True)

        # 6. Process and print the results for standings
        print(f"\nFound {len(matches_df_for_calc)} regular season match(es) to analyze for standings based on home/away rules:")
        display_columns = ['date_time_utc', 'home_team', 'home_score', 'away_score', 'away_team']
        print(matches_df_for_calc[display_columns].sort_values(by='date_time_utc').to_string(index=False))

        # 7. Calculate stats (GP, W, L, D, Points, GS, GC, GD)
        stats = {team_name: {'games_played': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'points': 0, 'goals_scored': 0, 'goals_conceded': 0} for team_name in team_ids.keys()}
        for _, row in matches_df_for_calc.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            home_score, away_score = row['home_score'], row['away_score']
            
            stats[home_team]['games_played'] += 1
            stats[away_team]['games_played'] += 1
            stats[home_team]['goals_scored'] += home_score
            stats[home_team]['goals_conceded'] += away_score
            stats[away_team]['goals_scored'] += away_score
            stats[away_team]['goals_conceded'] += home_score

            if home_score > away_score:
                stats[home_team]['points'] += 3
                stats[home_team]['wins'] += 1
                stats[away_team]['losses'] += 1
            elif away_score > home_score:
                stats[away_team]['points'] += 3
                stats[away_team]['wins'] += 1
                stats[home_team]['losses'] += 1
            else:
                stats[home_team]['points'] += 1
                stats[away_team]['points'] += 1
                stats[home_team]['draws'] += 1
                stats[away_team]['draws'] += 1

        # 8. Implement Tie-Breaker Logic
        table_data = [{'team_name': name, 'games_played': d['games_played'], 'wins': d['wins'], 'losses': d['losses'], 'draws': d['draws'], 'points': d['points'], 'goals_scored': d['goals_scored'], 'goals_conceded': d['goals_conceded'], 'goal_differential': d['goals_scored'] - d['goals_conceded'], 'h2h_points': 0} for name, d in stats.items()]
        
        points_groups = defaultdict(list)
        for team_data in table_data:
            points_groups[team_data['points']].append(team_data['team_name'])

        for _, tied_teams in points_groups.items():
            if len(tied_teams) > 1:
                tied_team_ids = [team_ids[name] for name in tied_teams]
                h2h_condition = (matches_df_for_calc['home_team_id'].isin(tied_team_ids)) & (matches_df_for_calc['away_team_id'].isin(tied_team_ids))
                h2h_matches = matches_df_for_calc[h2h_condition]
                h2h_stats = {name: {'points': 0} for name in tied_teams}
                for _, row in h2h_matches.iterrows():
                    if row['home_score'] > row['away_score']: h2h_stats[row['home_team']]['points'] += 3
                    elif row['away_score'] > row['home_score']: h2h_stats[row['away_team']]['points'] += 3
                    else:
                        h2h_stats[row['home_team']]['points'] += 1
                        h2h_stats[row['away_team']]['points'] += 1
                for team_data in table_data:
                    if team_data['team_name'] in h2h_stats:
                        team_data['h2h_points'] = h2h_stats[team_data['team_name']]['points']

        # 9. Sort the table and print
        sorted_table = sorted(table_data, key=lambda x: (x['points'], x['h2h_points'], x['goal_differential'], x['goals_scored']), reverse=True)
        print(f"\n--- {conference_name} Head-to-Head Table (Regular Season Only) ---")
        header = "{:<5} {:<25} {:>3} {:>3} {:>3} {:>3} {:>4} {:>4} {:>4} {:>4}".format("Pos", "Team", "GP", "W", "L", "D", "Pts", "GS", "GC", "GD")
        print(header)
        print("-" * len(header))
        for i, team in enumerate(sorted_table, 1):
            row = "{:<5} {:<25} {:>3} {:>3} {:>3} {:>3} {:>4} {:>4} {:>4} {:>4}".format(
                i, team['team_name'], team['games_played'], team['wins'], team['losses'], team['draws'],
                team['points'], team['goals_scored'], team['goals_conceded'], team['goal_differential']
            )
            print(row)
        
        standings_df = pd.DataFrame(sorted_table)
        standings_df = standings_df[['team_name', 'games_played', 'wins', 'losses', 'draws', 'points', 'goals_scored', 'goals_conceded', 'goal_differential']]
        
        top_team = sorted_table[0]['team_name'] if sorted_table else None
        return top_team, standings_df, all_conference_matches

    except Exception as e:
        print(f"\nAn unexpected error occurred in {conference_name} analysis: {e}")
        return None, pd.DataFrame(), pd.DataFrame()

def find_most_recent_winner(team1_name, team2_name, league, start_season, all_teams, asa_client):
    """
    Finds the most recent regular season match with a winner between two teams.

    Returns:
        dict: A dictionary with the match details, or None.
    """
    print(f"\n{'='*20} FINDING MOST RECENT WINNER {'='*20}")
    print(f"Searching for decisive match between {team1_name} and {team2_name}...")

    try:
        team1_id = all_teams[all_teams['team_name'] == team1_name]['team_id'].iloc[0]
        team2_id = all_teams[all_teams['team_name'] == team2_name]['team_id'].iloc[0]

        for year in range(int(start_season), 1995, -1):
            print(f"Checking {year} season...")
            games_this_year = pd.DataFrame(asa_client.get_games(leagues=[league], seasons=[str(year)], stages=['Regular Season']))
            
            if 'date_time_utc' not in games_this_year.columns and 'date' in games_this_year.columns:
                games_this_year.rename(columns={'date': 'date_time_utc'}, inplace=True)
            elif 'date_time_utc' not in games_this_year.columns:
                print(f"Warning: No date column found for {year}. Skipping.")
                continue

            condition1 = (games_this_year['home_team_id'] == team1_id) & (games_this_year['away_team_id'] == team2_id)
            condition2 = (games_this_year['home_team_id'] == team2_id) & (games_this_year['away_team_id'] == team1_id)
            matches_df = games_this_year[condition1 | condition2].copy()

            if not matches_df.empty:
                matches_df = matches_df.sort_values(by='date_time_utc', ascending=False)
                for _, row in matches_df.iterrows():
                    # Ensure the match has a score before checking for a winner
                    if pd.notna(row['home_score']) and pd.notna(row['away_score']) and row['home_score'] != row['away_score']:
                        home_team_name = all_teams[all_teams['team_id'] == row['home_team_id']]['team_name'].iloc[0]
                        away_team_name = all_teams[all_teams['team_id'] == row['away_team_id']]['team_name'].iloc[0]
                        winner = home_team_name if row['home_score'] > row['away_score'] else away_team_name
                        
                        print("\n--- Most Recent Decisive Match Found ---")
                        print(f"Date: {pd.to_datetime(row['date_time_utc']).strftime('%Y-%m-%d')}")
                        print(f"Match: {home_team_name} vs. {away_team_name}")
                        print(f"Score: {int(row['home_score'])} - {int(row['away_score'])}")
                        print(f"Winner: {winner}")
                        
                        match_data = {
                            'Date': pd.to_datetime(row['date_time_utc']).strftime('%Y-%m-%d'),
                            'Match': f"{home_team_name} vs. {away_team_name}",
                            'Score': f"{int(row['home_score'])} - {int(row['away_score'])}",
                            'Winner': winner
                        }
                        return match_data

        print(f"\nNo regular season match with a winner found between {team1_name} and {team2_name} since 1996.")
        return None

    except Exception as e:
        print(f"\nAn error occurred while finding the most recent winner: {e}")
        return None

def process_season(season, league, eastern_teams, western_teams, all_teams_df, client):
    """
    Processes a single season's data.
    """
    SEASON = str(season)
    print(f"\n{'#'*40}\n# PROCESSING SEASON: {SEASON}\n{'#'*40}")
    
    os.makedirs(SEASON, exist_ok=True)
    print(f"Output directory '{SEASON}/' is ready.")

    _, eastern_standings, eastern_all_matches = calculate_h2h_points(eastern_teams, league, SEASON, "Eastern", all_teams_df, client)
    _, western_standings, western_all_matches = calculate_h2h_points(western_teams, league, SEASON, "Western", all_teams_df, client)

    # Export conference data to CSV in the year's subfolder
    if eastern_standings is not None and not eastern_standings.empty:
        filepath = os.path.join(SEASON, f'eastern_standings_{SEASON}.csv')
        eastern_standings.to_csv(filepath, index=False)
        print(f"\nEastern conference standings saved to {filepath}")
    if eastern_all_matches is not None and not eastern_all_matches.empty:
        filepath = os.path.join(SEASON, f'eastern_matches_{SEASON}.csv')
        eastern_all_matches.to_csv(filepath, index=False)
        print(f"All Eastern conference matches saved to {filepath}")
        
    if western_standings is not None and not western_standings.empty:
        filepath = os.path.join(SEASON, f'western_standings_{SEASON}.csv')
        western_standings.to_csv(filepath, index=False)
        print(f"Western conference standings saved to {filepath}")
    if western_all_matches is not None and not western_all_matches.empty:
        filepath = os.path.join(SEASON, f'western_matches_{SEASON}.csv')
        western_all_matches.to_csv(filepath, index=False)
        print(f"All Western conference matches saved to {filepath}")

    # Find and export decisive matches
    print(f"\n{'='*20} ALL EAST VS. WEST DECISIVE MATCHUPS ({SEASON}) {'='*20}")
    decisive_matches_list = []
    if eastern_teams and western_teams:
        for east_team in eastern_teams:
            for west_team in western_teams:
                if east_team in all_teams_df['team_name'].values and west_team in all_teams_df['team_name'].values:
                    result = find_most_recent_winner(east_team, west_team, league, SEASON, all_teams_df, client)
                    if result:
                        decisive_matches_list.append(result)
                else:
                    print(f"\nSkipping matchup due to one or more invalid team names: '{east_team}' vs '{west_team}'")
    
    if decisive_matches_list:
        decisive_df = pd.DataFrame(decisive_matches_list)
        filepath = os.path.join(SEASON, f'decisive_matches_{SEASON}.csv')
        decisive_df.to_csv(filepath, index=False)
        print(f"\nAll decisive matches saved to {filepath}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <year> or 'all-years'")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    LEAGUE = "mls"
    eastern_teams = ["New England Revolution", "New York Red Bulls", "D.C. United", "Columbus Crew", "Chicago Fire FC"]
    western_teams = ["LA Galaxy", "San Jose Earthquakes", "Colorado Rapids", "Sporting Kansas City", "FC Dallas"]

    try:
        client = AmericanSoccerAnalysis()
        all_teams_df = pd.DataFrame(client.get_teams(leagues=[LEAGUE]))

        if not all_teams_df.empty:
            if arg.lower() == 'all-years':
                for year in range(2025, 2014, -1):
                    process_season(year, LEAGUE, eastern_teams, western_teams, all_teams_df, client)
            else:
                try:
                    year_arg = int(arg)
                    if not (2013 <= year_arg <= 9999):
                        raise ValueError("Year out of range.")
                    process_season(year_arg, LEAGUE, eastern_teams, western_teams, all_teams_df, client)
                except ValueError:
                    print("Error: Please provide a valid 4-digit year (e.g., 2023) or 'all-years'.")
                    sys.exit(1)

    except Exception as e:
        print(f"A critical error occurred: {e}")
