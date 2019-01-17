import pandas as pd
import os

refs_home = {
    'abs': 49,
    'hits': 50,
    'doubles': 51,
    'triples': 52,
    'hrs': 53,
    'rbis': 54,
    'sac_hits': 55,
    'sac_flies': 56,
    'hbp': 57,
    'bb': 58,
    'bbi': 59,
    'ko': 60,
    'sb': 61,
    'cs': 62,
    'hit_into_dp': 63,
    'catcher_i': 64,
    'lob': 65,
    'pitchers': 66,
    'ers': 68,
    'wp': 69,
    'balks': 70,
    'assists': 72,
    'errors': 73,
    'passed_balls': 74,
    'dp': 75,
    'tp': 76
}

refs_away = {
    'abs': 21,
    'hits': 22,
    'doubles': 23,
    'triples': 24,
    'hrs': 25,
    'rbis': 26,
    'sac_hits': 27,
    'sac_flies': 28,
    'hbp': 29,
    'bb': 30,
    'bbi': 31,
    'ko': 32,
    'sb': 33,
    'cs': 34,
    'hit_into_dp': 35,
    'catcher_i': 36,
    'lob': 37,
    'pitchers': 38,
    'ers': 40,
    'wp': 41,
    'balks': 42,
    'assists': 44,
    'errors': 45,
    'passed_balls': 46,
    'dp': 47,
    'tp': 48
}


def get_all_last_n(df, index, n):
    all_last_n = {}
    for key, val_h in refs_home.items():
        val_a = refs_away[key]
        ht_lastn_h, ht_lastn_a = get_last_n_home(df, index, val_h, val_a, n)
        all_last_n["ht_l" + str(n) + "_" + key + "_h"] = ht_lastn_h
        all_last_n["ht_l" + str(n) + "_" + key + "_a"] = ht_lastn_a
    for key, val_h in refs_home.items():
        val_a = refs_away[key]
        at_lastn_h, at_lastn_a = get_last_n_away(df, index, val_h, val_a, n)
        all_last_n["ht_l" + str(n) + "_" + key + "_h"] = at_lastn_h
        all_last_n["ht_l" + str(n) + "_" + key + "_a"] = at_lastn_a
    return all_last_n


def generate_data(df, n):
    df_data = {}
    counter = 0
    for index, row in df.iterrows():
        ht_hw, ht_aw = get_home_team_wins(df, index)
        ht_wins = len(ht_hw.index) + len(ht_aw.index)
        ht_hl, ht_al = get_home_team_losses(df, index)
        ht_losses = len(ht_hl.index) + len(ht_al.index)

        at_hw, at_aw = get_away_team_wins(df, index)
        at_wins = len(at_hw.index) + len(at_aw.index)
        at_hl, at_al = get_away_team_losses(df, index)
        at_losses = len(at_hl.index) + len(at_al.index)

        if df.at[index, 9] < df.at[index, 10]:
            home_team_won = True
        elif df.at[index, 9] > df.at[index, 10]:
            home_team_won = False
        else:
            home_team_won = None

        ht = df.at[index, 6]
        at = df.at[index, 3]

        row = get_all_last_n(df, index, n)
        row["ht"] = ht
        row["ht_wins"] = ht_wins
        row["ht_losses"] = ht_losses
        row["at"] = at
        row["at_wins"] = at_wins
        row["at_losses"] = at_losses
        row["home_team_won"] = home_team_won

        for key, val in row.items():
            try:
                df_data[key].append(val)
            except KeyError:
                df_data[key] = [val]

        counter += 1
        print(counter)
        print(row)
    data = pd.DataFrame(df_data)
    return data


# returns home team's wins at home and away
def get_home_team_wins(df, index):
    team = df.at[index, 6]
    print(team)
    home_wins = df.loc[(df.index < index) &
                  (df[6] == team) &
                  (df[10] > df[9])]
    away_wins = df.loc[(df.index < index) &
                       (df[3] == team) &
                       (df[10] < df[9])]
    return home_wins, away_wins


# returns home team's losses at home and away
def get_home_team_losses(df, index):
    team = df.at[index, 6]
    home_losses = df.loc[(df.index < index) &
                    (df[6] == team) &
                    (df[10] < df[9])]
    away_losses = df.loc[(df.index < index) &
                         (df[3] == team) &
                         (df[10] > df[9])]
    return home_losses, away_losses


# returns away team's wins on road and at home
def get_away_team_wins(df, index):
    team = df.at[index, 3]
    home_wins = df.loc[(df.index < index) &
                       (df[6] == team) &
                       (df[10] > df[9])]
    away_wins = df.loc[(df.index < index) &
                  (df[3] == team) &
                  (df[10] < df[9])]
    return home_wins, away_wins


# returns away team's losses on road and at home
def get_away_team_losses(df, index):
    team = df.at[index, 3]
    home_losses = df.loc[(df.index < index) &
                         (df[6] == team) &
                         (df[10] < df[9])]
    away_losses = df.loc[(df.index < index) &
                    (df[3] == team) &
                    (df[10] > df[9])]
    return home_losses, away_losses


# returns the sum of a given stat over the home team's last n home and away games
def get_last_n_home(df, index, attr_h, attr_a, n):
    team = df.at[index, 6]
    all_prior_home = df.loc[(df.index < index) & (df[6] == team)]
    last_n_home = all_prior_home.tail(n)
    total_home = last_n_home[attr_h].sum()

    all_prior_away = df.loc[(df.index < index) & (df[3] == team)]
    last_n_away = all_prior_away.tail(n)
    total_away = last_n_away[attr_a].sum()
    return total_home, total_away


# returns the sum of a given stat over the away team's last n home and away games
def get_last_n_away(df, index, attr_h, attr_a, n):
    team = df.at[index, 3]
    all_prior_home = df.loc[(df.index < index) & (df[6] == team)]
    last_n_home = all_prior_home.tail(n)
    total_home = last_n_home[attr_h].sum()

    all_prior_away = df.loc[(df.index < index) & (df[3] == team)]
    last_n_away = all_prior_away.tail(n)
    total_away = last_n_away[attr_a].sum()
    return total_home, total_away


def save_to_csv(filename, data):
    data.to_csv(filename)


def get_home_streak(df, team, game_index):
    streak = df.loc[(df.index < game_index)]


def get_games_by_team(df, team):
    team_data = df.loc[(df[3] == team) | (df[6] == team)]
    return team_data


def get_columns_to_sum():
    cols_to_sum = [9, 10]
    cols_to_sum += list(range(21, 77))
    cols_to_sum.remove(39)
    return cols_to_sum


def main():
    pd.set_option('display.max_columns', 500)
    dfs = []

    for filename in os.listdir("./data/raw"):
        print(filename)
        df = pd.read_csv(f"./data/raw/{filename}", header=None)

        remove = [1, 4, 7, 11, 13, 14, 15, 16, 17, 18, 19, 20, 39]
        for i in range(77, 161):
            remove.append(i)

        df.drop(remove, 1, inplace=True)

        data = generate_data(df, 10)
        print(data.head(10))
        dfs.append(data)

    all_data = pd.concat(dfs, ignore_index=True)
    save_to_csv("./data/all_master_data.csv", all_data)


if __name__ == "__main__":
    main()