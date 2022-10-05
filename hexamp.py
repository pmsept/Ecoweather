

import pandas as pd
import math


# Change these to get infomation from a excel document
FILENAME = 'final_observations_rainfall_daily_1900-2005.xlsx'
SHEET = 0
KEEPCOL = ['ons_id', 'constituency', 'pr_ob']
numrange = 0.9
numoffset = -0.04

# Dictionary for the rgb values of each political party
rgbdict = {
    "Lab": "rgb(228,0,59)",
    "Con": "rgb(0,135,220)",
    "SNP": "rgb(253,243,142)",
    "Green": "rgb(106,176,35)",
    "LD": "rgb(253,187,48)",
    "PC": "rgb(252,244,3)",
    "DUP": "rgb(212,105,76)",
    "SF": "rgb(3,97,47)",
    "SDLP": "rgb(82,5,7)",
    "Spk": "rgb(182,217,217)",
    "Alliance": "rgb(176,170,5)"}

data = {}

if __name__ == "__main__":

    # Opens the excel file and saves it as a DataFrame
    df = pd.read_excel(FILENAME, SHEET)

    if not df.empty:
        # Removes all unnesesary columns from the DataFrame
        dropcol = df.columns
        for i in range(dropcol.size):
            if dropcol[i] not in KEEPCOL:
                df = df.drop(columns=[dropcol[i]])

        # Puts all the necessary data into a dictionary
        for row in df.itertuples():
            # example: rgb = "rgb(" + str(math.trunc((255 / numrange) * (row[3] - numoffset))) + ",0," + str(math.trunc(255 - ((255 / numrange) * (row[3] - numoffset))))

            #blue
            rgb = "rgb(" + str(math.trunc(255-((255 / numrange) * (row[3] - numoffset)))) +","+ str(math.trunc(255-((255 / numrange) * (row[3] - numoffset)))) +",255)"

            #red
            #rgb = "rgb(255," + str(math.trunc(255 - ((255 / numrange) * (row[3] - numoffset)))) + "," + str(math.trunc(255 - ((255 / numrange) * (row[3] - numoffset)))) + ")"

            # Adds an RGB value depending on the political party
            #  if row[3] in rgbdict.keys():
            #   rgb = rgbdict[row[3]]
            #  else:
            #    rgb = "rgb(255,255,255)"

            data[row[1]] = [row[2], round(row[3],3), rgb]

        goodfile = open('hexjson.txt', 'a')
        goodfile.truncate(0)

        with open('constituencies.txt') as file:
            for line in file:
                newline = line
                if line.endswith("},\n"):
                    conid = line[3:12]
                    if conid in data.keys():
                        newline = line[:-3] + ''',"pop":"''' + str(data[conid][1]) + '''","colour":"''' + data[conid][
                        2] + '''"},\n'''
                    else:
                        newline = ""
                goodfile.write(newline)

        goodfile.close()

