import pandas as pd
from processing.data_handling import save_to_csv


def get_games_by_team(df, team):
    games = df.loc[(df[1] == team) | (df[10] == team)]
    return games


def compare_values(df, team, values, expected):
    home_games = df.loc[df[1] == team]
    away_games = df.loc[df[10] == team]

    home_val = values[0]
    away_val = values[1]
    expected = expected.drop(expected.index[-1])
    expected = expected.rename("values")

    original_home = home_games[home_val]
    original_away = away_games[away_val]
    original = pd.concat([original_home, original_away]).sort_index()
    original = original.rename("values")
    original = original.drop(original.index[0])
    counter = 0
    for index, value in original.iteritems():
        original = original.replace(value, int(value))
        original = original.rename({index: counter})
        counter += 1
    comparison = original == expected
    print(original)
    print(expected)
    return comparison


def main():
    pd.set_option('display.max_columns', 500)
    df = pd.read_csv("./data/data.csv", header=None)
    ari_gbg = pd.read_csv("./data/ari_wins.csv")

    ari_games = get_games_by_team(df, "ARI")
    ari_wins = ari_gbg["W-L"]
    for index, value in ari_wins.iteritems():
        ari_wins.replace(value, int(value.split("-")[1]), inplace=True)

    win_comparison = compare_values(ari_games, "ARI", [3, 12], ari_wins)
    print(win_comparison)
    save_to_csv("./data/ari_games.csv", ari_games)

if __name__ == "__main__":
    main()

