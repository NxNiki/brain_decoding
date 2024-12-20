import numpy as np

SF = 2000
TWILIGHT_ANNOTATION_FS = 4
MOVIE24_ANNOTATION_FS = 1
PREDICTION_FS = 4

OFFSET = {
    "555": 4.58,
    "562": 134.194,
    "564": [(792.79, 1945), (3732.44, 5091)],
    "566": 380.5814,
}
CONTROL = {
    "566": [(121, 1520), (1544, 2825)],
}
SPONTANEOUS_SAMPLES = {
    "555": 600000,
    "562": 550000,
    "564": [
        (int(417.79 * SF), int(777.79 * SF)),
        (int(3357.44 * SF), int(3717.44 * SF)),
    ],
    "566": 360 * SF,
}
SPONTANEOUS_FRAMES = {
    "555": 600000,
    "562": 550000,
    "564": [
        (int(417.79 * 30), int(777.79 * 30)),
        (int(3357.44 * 30), int(3717.44 * 30)),
    ],
}
SPIKE_CHANNELS = {
    "555": list(np.arange(1, 25)) + list(np.arange(41, 46)) + [47, 66, 68, 74] + list(np.arange(80, 89)),
    "562": list(np.arange(1, 25)) + list(np.arange(34, 49)) + list(np.arange(65, 72)) + list(np.arange(73, 81)),
    "564": [2, 3]
    + list(np.arange(5, 10))
    + list(np.arange(11, 17))
    + [34, 36, 39, 41, 42, 44, 45, 47]
    + list(np.arange(49, 65))
    + [69, 76, 77],
}
SPIKE_WINDOWS = {
    "555": [(0, 24), (2, 26), (4, 28), (6, 30)],
    "562": [(0, 8), (7, 15), (15, 23), (22, 30)],
    "564": [(0, 8), (7, 15), (15, 23), (22, 30)],
}

SPIKE_CHANNEL = {
    "555": 37,
    "562": 56,
    "563": 64,
    "564": 56,
    "565": 80,
    "566": 72,
    "567": 40,
    "568": 80,
    "570": 80,
    "572": 72,
    "i728": 96,
}  # "565": 80, '566': 72

SPIKE_FRAME = {
    "555": 24,
    "562": 50,
    "563": 50,
    "564": 50,
    "565": 50,
    "566": 50,
    "567": 50,
    "568": 50,
    "570": 50,  # unverified.
    "572": 50,
    "i728": 50,
}  # 8, 15, 24

LFP_CHANNEL = {
    "555": 208,
    "562": 504,
    "563": 624,
    "564": 10 * 2 * 8,
    "565": 11 * 16,
    "566": 568,
    "567": 768,
    "568": -1,
    "i728": 912,
}  # 11*2*8  568

LFP_FRAME = {
    "555": 208,
    "562": 500,
    "563": 500,
    "564": 500,
    "565": 500,
    "566": 500,
    "567": 500,
    "568": 500,
    "i728": 500,
}
# LABELS = ['LosAngeles', 'BombAttacks', 'Whitehouse', 'CIA/FBI', 'Hostage', 'Handcuff', 'Jack', 'Chloe', 'Bill', 'A. Fayed', 'A. Amar', 'President']
LABELS = [
    "WhiteHouse",
    "CIA",
    "Hostage",
    "Handcuff",
    "Jack",
    "Bill",
    "A. Fayed",
    "A. Amar",
]

TWILIGHT_LABELS = [
    "Alice.Cullen",
    "Angela.Weber",
    "Bella.Swan",
    "Billy.Black",
    "Carlisle.Cullen",
    "Charlie.Swan",
    "Edward.Cullen",
    "Emmett.Cullen",
    "Eric.Yorkie",
    "Jacob.Black",
    "Jasper.Hale",
    "Jessica.Stanley",
    "Mike.Newton",
    "No.Characters",
    "Rosalie.Hale",
    "Side.Character",
    "Renee.Swan",
    "Tyler.Crowley",
]


TWILIGHT_LABELS_MERGE = ["Bella.Swan", "Edward.Cullen", "No.Characters", "Others"]  # "Others" must be the last element.
TWILIGHT_24_LABELS = ["Twilight", "24"]
