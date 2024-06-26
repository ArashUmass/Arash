import math
import csv
from copy import deepcopy
from calendar import isleap
from symfit import parameters, variables, sin, cos, Fit
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor

import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------------------------

number_of_cosines = 1
name_of_day = ""

number_of_turbines = 1446
# database variables
fday=1/24
fyear=1/(24*365)
# Offshore:
csvData = f"Net Demand Station 44025 - 2006 to 2020 - {number_of_turbines}T - Copy.csv"
# csvData = f"Net Demand Station 44025 - 2006 to 2020 - {number_of_turbines}T - Copy.csv"

# Onshore:
# csvData = f"Net Demand Onshore Newark - 2006 to 2020 - {number_of_turbines}T.csv"

dataType = "Offshore"
# dataType = "Onshore"

# DO NOT MODIFY THE FOLLOWING VALUES
quantiles = []
qyDict = {}
qyInfo = {}
nanCount = 0
previousBin = -1
binResults = None
binDates = None

monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# air density (kg/m^3)
rho = 1.225

# turbine blade radius: 120m
tbRadius = 120

minSpeed = 3
maxSpeed = 10.59

alpha = 0.1
h0 = 10
hHub = 150

# MODIFY IF NEEDED
windSpeedLimit = [-0.2, 20]
windSpeedLimit = None

windPowerLimit = [-15000, 30000]
# windPowerLimit = None

colors = ["orange", "DodgerBlue", "red", "chocolate", "lightslategray", "magenta", "midnightblue", "darkkhaki"]
column = 9
transitionCount = 5

f = 1/365
w = f*2*np.pi

# -------------------------

# initial commands

with open(csvData, "r") as file:
	fLines = file.readlines()

# -------------------------------

# functions

def getImageName(date, fourier = False, linear = False, poly = False, power = False,
								 n = number_of_cosines, endDate = None, plotType = "plotDay", hourly = True,
								 quantile = False, q = None):

	if not power:
		ans = f"Net Demand - {number_of_turbines}T "

	else:
		ans = "Wind Power - "

	if plotType == "plotRange":
		ans += "Range "

	elif plotType == "plotTimeInterval":
		ans += "Interval "

	elif plotType == "plotGeoff":
		ans += "Geoff "

	elif plotType == "plotGolbon":
		ans += "Golbon "

	elif plotType == "plotAnnual":
		ans += "Annual "

	ans += "{}".format(date)

	if endDate:
		ans += " to " + endDate

	if fourier or linear or poly or quantile or plotType == "plotGeoff":
		ans += " -"

	if fourier or quantile or plotType == "plotGeoff":
		ans += " n = {}".format(n)

		if linear or poly or hourly:
			ans += ","

	if linear:
		ans += " Linear"

	if poly:
		ans += " Poly"

	if quantile:
		ans += " Quantile"

	return ans


def datify(text):

	return text[0:4] + "-" + text[4:6] + "-" + text[6:]


def subtractLists(a, b):

	if len(a) != len(b):
		return []

	ans = []
	
	for i, j in zip(a, b):
		ans.append(i - j)

	return ans


def multiplyListByNumber(listy, n):

	for i in range(len(listy)):
		listy[i] *= n

	return listy


def extractIndex(listy, index, toFloat = False, toInt = False):
	
	ans = []

	for i in listy:
		if toFloat:
			ans.append(float(i[index]))

		elif toInt:
			ans.append(int(i[index]))

		else:
			ans.append(i[index])

	return ans


def convertWindSpeed(v):

	return v * (hHub / h0) ** alpha


def calculateWindPower(v):

	return (16/27) * 0.5 * rho * math.pi * (tbRadius ** 2) * ((v-3) ** 3) * (hHub / h0) ** (3 * alpha) / 1_000_000


def defaultMatrix(rows, columns, value = 0):

	ans = []

	for i in range(rows):

		if type(value) == list:
			temp = [deepcopy(value) for j in range(columns)]

		else:
			temp = [value for j in range(columns)]

		ans.append(temp)

	return ans


def getDayInfo(date, display = True):

	listy = []

	for i in fLines:

		j = i.split(",")
		j[-1] = j[-1][:-1]

		if j[0] + j[1] + j[2] == date:

			if j[4] == "50":
				if display:
					print(j)
					print()

				listy.append(j)

	# print("{} => {}".format(date, str(len(listy))))

	return listy


def saveImage(name):

	plt.savefig(name + ".png")


def getRangeDates(startingDate, endDate):

	limit = 400000
	count = 0
	listy = []

	year = int(startingDate[0:4])
	month = int(startingDate[4:6])
	day = int(startingDate[6:8]) - 1

	startingYear = int(startingDate[0:4])
	startingMonth = int(startingDate[4:6])
	startingDay = int(startingDate[6:8])

	targetYear = int(endDate[0:4])
	targetMonth = int(endDate[4:6])
	targetDay = int(endDate[6:8])

	if isleap(year):
		monthDays[1] = 29

	while (month != targetMonth or day != targetDay or year != targetYear) and count < limit:

		if day != monthDays[month - 1]:
			day += 1

		else:
			if month != 12:
				day = 1
				month += 1

			else:
				day = 1
				month = 1
				year += 1

				if isleap(year):
					monthDays[1] = 29

				else:
					monthDays[1] = 28

		ans = str(year)

		if month < 10:
			ans += "0"

		ans += str(month)

		if day < 10:
			ans += "0"

		ans += str(day)
		listy.append(ans)

		count += 1

	return listy


def getRangeDayInfo(startingDate, endDate, display = True):

	yList = []
	days = getRangeDates(startingDate, endDate)

	for i in days:
		listy = getDayInfo(i, display = display)

		for j in listy:
			yList.append(float(j[column]))

	return yList


def fourier_series(x, f, mean, n = 3):
	"""
	Returns a symbolic fourier series of order `n`.

	:param n: Order of the fourier series.
	:param x: Independent variable
	:param f: Frequency of the fourier series
	"""
	# Make the parameter objects for all the terms

	a0, *cos_a = parameters(",".join(["a{}".format(i) for i in range(0, n + 1)]))
	sin_b = parameters(",".join(["b{}".format(i) for i in range(1, n + 1)]))
	# Construct the series
	series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
						 for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start = 1))

	print("series:")
	print(series)
	print()

	return series



def linearReg(x, y, show = False):

	x = x.reshape(-1, 1)

	model = LinearRegression().fit(x, y)

	r_sq = model.score(x, y)
	y_pred = model.predict(x)
	y_pred = model.intercept_ + model.coef_ * x

	if show:
		print(f"coefficient of determination: {r_sq}")
		print(f"intercept: {model.intercept_}")
		print(f"slope: {model.coef_}")
		print(f"predicted response:\n{y_pred}")
		print(f"predicted response:\n{y_pred}")

	plt.plot(x, y_pred, "r-", label = "Linear Reg")


def polyReg(x, y, show = False):

	x = x.reshape(-1, 1)
	x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
	model = LinearRegression().fit(x_, y)
	
	r_sq = model.score(x_, y)
	y_pred = model.predict(x_)

	if show:
		print(f"coefficient of determination: {r_sq}")
		print(f"intercept: {model.intercept_}")
		print(f"coefficients: {model.coef_}")
		print(f"predicted response:\n{y_pred}")

	plt.plot(x, y_pred, "g-", label = "Poly Reg")


def getQuantile(x, y, quantiles, qyDict):

	ans = []
	precision = 6
	# y = round(float(y), 1)

	if str(y).lower() == "nan":
		# text += " => NaN"
		# print("gqyZ: NaN => None")
		return None

	y = np.float32(y)
	qCount = 0

	for q in quantiles:

		if y <= qyDict[q][x]:
			# print(f"gqy: {y} => {qCount}, qyDict[{q}]: {qyDict[q][x]}")
			return qCount

		qCount += 1

	print(f"gqyZ: {y} => {qCount}, qyDict: {qyDict[list(qyDict)[-1]][x]}")
	return qCount


def qyDictInfo(countTable):

	if qyDict:

		i = 0
		totalSum = 0
		quantiles = list(qyDict)
		print("quantiles:", quantiles)
		
		for i in range(len(countTable)):

			row = countTable[i]
			tempSum = sum(countTable[i])
			qyInfo[i] = tempSum
			# qyDictInfo[q] = sum()
			totalSum += tempSum
			# i += 1

		finalProb = 0

		for i in range(len(countTable)):
			qyInfo[i] = round(qyInfo[i] / totalSum, 6)
			finalProb += qyInfo[i]

		print()
		print(qyInfo)
		print("Probability Sum:", finalProb)
		print()


def printer(listy, inline = True):

	for i in listy:

		for j in range(len(i)):

			print(i[j], end = "")

			if j != len(i) - 1:
				print(", ", end = "")

		print()


def getMarkovTable(yq, q):

	global previousBin

	print("_____________________________")
	print(f"yq: {yq}, q: {q}")
	# print("qyDict:")
	# print(qyDict)
	print()

	ans = defaultMatrix(q + 1, q + 1)

	for i in range(len(yq) - 1):

		a = yq[i]
		b = yq[i + 1]

		aStr = str(a)[0]
		bStr = str(b)[0]

		# print(f"{aStr} => {bStr}")

		if a != None and b != None:
			ans[a][b] += 1
			# print("PASS")

		# print("-----------------------")

	probAns = deepcopy(ans)

	for i in range(len(probAns)):

		rowSum = sum(probAns[i])

		for j in range(len(probAns[i])):

			if rowSum > 0:
				probAns[i][j] = round(probAns[i][j] / rowSum, 2)

	return ans, probAns


def writeMarkovTable(table, tableType, dateStr, q, hourly = True):

	name = f"ND Markov Table - {dateStr} - {q}Q, n = {number_of_cosines}"

	if hourly:
		name += ", Hourly"

	name += f" - {tableType}.csv"

	with open(name, "w", newline = "") as f:

		writer = csv.writer(f)
		writer.writerow(["n"] + list(range(q + 1)))

		count = 0

		for i in table:

			writer.writerow([count] + i)
			count += 1


def quantileReg(x, y, date, endDate = None, q = 0.5, markov = False, save = False, zoom = False,
								hourly = True, limit = None, qcsv = False, summary = True, qlw = 1):
	
	global qyDict, quantiles

	print()
	print("len(y):", len(y))

	df = pd.DataFrame({"xdata": x, "ydata": y})

	if zoom:
		print("ZOOM!")
		offset = 2
		# plt.gca().set_ylim([min(y) - offset, max(y) + offset])

	a0 = np.mean(y)

	'''
	formula = f"""ydata ~ 1 + I(np.cos(1 * 2 * np.pi * xdata * {fyear})) + I(np.sin(1 * 2 * np.pi * xdata * {fyear}))
				+ I(np.cos(2 * 2 * np.pi * xdata * {fyear})) + I(np.sin(2 * 2 * np.pi * xdata * {fyear}))"""
	'''
	'''
	formula = f"""ydata ~ 1 + I(np.cos(1 * 2 * np.pi * xdata * {fyear})) + I(np.sin(1 * 2 * np.pi * xdata * {fyear})) +
	I(np.cos(2 * 2 * np.pi * xdata * {fyear})) + I(np.sin(2 * 2 * np.pi * xdata * {fyear})) +
	I(np.cos(3 * 2 * np.pi * xdata * {fyear})) + I(np.sin(3 * 2 * np.pi * xdata * {fyear}))"""
	print(f"?{formula}?")
	'''

	# n=1
	'''
	formula = f"""ydata ~ 1 + I(np.cos(1 * 2 * np.pi * xdata * {fday})) + I(np.sin(1 * 2 * np.pi * xdata * {fday})) +
			  I(np.cos(1 * 2 * np.pi * xdata * {fyear})) + I(np.sin(1 * 2 * np.pi * xdata * {fyear})) +
			  I(np.cos(1 * 2 * np.pi * xdata * {fday})*np.cos(1 * 2 * np.pi * xdata * {fyear})) + 
			  I(np.cos(1 * 2 * np.pi * xdata * {fday})*np.sin(1 * 2 * np.pi * xdata * {fyear})) +
			  I(np.sin(1 * 2 * np.pi * xdata * {fday})*np.cos(1 * 2 * np.pi * xdata * {fyear})) +
			  I(np.sin(1 * 2 * np.pi * xdata * {fday})*np.sin(1 * 2 * np.pi * xdata * {fyear}))"""
	'''
		# n=2
	
	formula = f"""ydata ~ 1 + I(np.cos(1 * 2 * np.pi * xdata * {fday})) + I(np.sin(1 * 2 * np.pi * xdata * {fday})) +
        I(np.cos(1 * 2 * np.pi * xdata * {fyear})) + I(np.sin(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.cos(1 * 2 * np.pi * xdata * {fday})*np.cos(1 * 2 * np.pi * xdata * {fyear})) + 
        I(np.cos(1 * 2 * np.pi * xdata * {fday})*np.sin(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(1 * 2 * np.pi * xdata * {fday})*np.cos(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(1 * 2 * np.pi * xdata * {fday})*np.sin(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.cos(2 * 2 * np.pi * xdata * {fday})) + I(np.sin(2 * 2 * np.pi * xdata * {fday})) +
        I(np.cos(2 * 2 * np.pi * xdata * {fyear})) + I(np.sin(2 * 2 * np.pi * xdata * {fyear})) +
        I(np.cos(1 * 2 * np.pi * xdata * {fday})*np.cos(2 * 2 * np.pi * xdata * {fyear})) + 
        I(np.cos(1 * 2 * np.pi * xdata * {fday})*np.sin(2 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(1 * 2 * np.pi * xdata * {fday})*np.cos(2 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(1 * 2 * np.pi * xdata * {fday})*np.sin(2 * 2 * np.pi * xdata * {fyear})) +
        I(np.cos(2 * 2 * np.pi * xdata * {fday})*np.cos(1 * 2 * np.pi * xdata * {fyear})) + 
        I(np.cos(2 * 2 * np.pi * xdata * {fday})*np.sin(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(2 * 2 * np.pi * xdata * {fday})*np.cos(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(2 * 2 * np.pi * xdata * {fday})*np.sin(1 * 2 * np.pi * xdata * {fyear})) +
        I(np.cos(2 * 2 * np.pi * xdata * {fday})*np.cos(2 * 2 * np.pi * xdata * {fyear})) + 
        I(np.cos(2 * 2 * np.pi * xdata * {fday})*np.sin(2 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(2 * 2 * np.pi * xdata * {fday})*np.cos(2 * 2 * np.pi * xdata * {fyear})) +
        I(np.sin(2 * 2 * np.pi * xdata * {fday})*np.sin(2 * 2 * np.pi * xdata * {fyear}))"""
	
	
	# for i in range(1, number_of_cosines + 1):

		# formula += f" + I(np.cos({i} * 2 * np.pi * xdata * {f})) + I(np.sin({i} * 2 * np.pi * xdata * {f}))"

	# print(formula)

	model = smf.quantreg(formula, df)
	# print(model)

	quantiles = [0.25, 0.5, 0.75]
	# quantiles = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

	qyDict = {}

	if not q in quantiles:
		quantiles.append(q)

	quantiles.sort()

	res_all = [model.fit(q = q) for q in quantiles]

	colorCount = 0
	summaryText = ""

	for qm, res in zip(quantiles, res_all):

		# print("qm:")
		# print(qm)
		# print()
		print(res.summary())
		summaryText += str(res.summary()) + "\n\n\n"
		print("*******************************")
		print()

		ls = "-"
		color = "k"

		if colorCount < len(colors):
			color = colors[colorCount]

		label = "Quantile {}".format(qm)
		yq = res.predict({"xdata": x})
		qyDict[qm] = yq
		lineWidth = qlw

		if colorCount <= len(colors):
			if limit:
				plt.plot(x[:limit], yq[:limit], linestyle = ls, lw = lineWidth, color = color, label = label)

			else:
				plt.plot(x, yq, linestyle = ls, lw = lineWidth, color = color, label = label)

			colorCount += 1

		else:
			if limit:
				plt.plot(x[:limit], yq[:limit], linestyle = ls, lw = lineWidth, label = label)
			else:
				plt.plot(x, yq, linestyle = ls, lw = lineWidth, label = label)

	# plt.legend()


	if summaryText:

		print("summary (save):", summary)
		fileName = f"Net Demand Summary - {number_of_turbines}T Hour {date} to {endDate}, n={number_of_cosines}.txt"

		with open(fileName, "w") as f:
			f.write(summaryText)


	if markov:

		number = 0
		yQuantiles = []
		overCount = 0
		estimatedOC = round((1 - quantiles[-1]) * len(y), 2)

		for i in y:

			temp = getQuantile(number, i, quantiles, qyDict)
			yQuantiles.append(temp)

			if temp == len(quantiles):
				overCount += 1

			number += 1

		# print("----------------")
		print(yQuantiles)
		print("overCount:", overCount)
		print("estimatedOC:", estimatedOC)

		countTable, probTable = getMarkovTable(yQuantiles, len(quantiles))
		# calcRunLength(date, endDate, yQuantiles, quantiles, save = True)

		dateStr = date
		qyDictInfo(countTable)

		print("endDateZ =>", endDate)

		if endDate:
			dateStr += f" to {endDate}"

		printer(countTable)

		writeMarkovTable(countTable, "Count", dateStr, len(quantiles), hourly = hourly)
		writeMarkovTable(probTable, "Probability", dateStr, len(quantiles), hourly = hourly)

	if qcsv and limit:

		qval = []
		firstRow = ["#"]
		firstRow.extend(list(qyDict))

		outputFile = f"Net Demand Quantiles - {number_of_turbines}T - {date} to {endDate}.csv"

		with open(outputFile, "w", newline = "") as file:

			writer = csv.writer(file)
			writer.writerow(firstRow)

			for i in range(limit):
				temp = [round(qyDict[q][i], 1) for q in qyDict]
				writer.writerow([i] + temp)


def binPrinter(save = False, saveName = "$"):

	global binResults, binDates

	maxNum = 0

	for i in binResults:
		tempMax = max(map(max, i))
		maxNum = max(maxNum, tempMax)

	maxNum = str(maxNum)
	numLength = len(maxNum)

	ans = ""
	length = len(binResults)

	for i in range(length):
		for j in range(length):

			tempString = []

			for k in range(transitionCount):
				tempDate = binDates[i][j][k]
				tempNum = str(binResults[i][j][k])

				if tempDate == "-":
					tempDate = " " * 12

				else:
					tempDate = "(" + tempDate + ")"

				tempString.append(f"{tempNum.ljust(numLength)} {tempDate}")

			tempString = " | ".join(tempString)
			ans += f"T {i} => {j}:  {tempString}\n"

		if i != length - 1:
			ans += "\n"

	print(ans)

	if save:
		saveName = saveName.replace("$", "ND Run Length")
		ans = saveName + "\n\n" + ans
		saveName += ".txt"

		with open(saveName, "w") as r:
			r.write(ans)

	return ans


def calcRunLength(startingDate, endDate, listy, quantiles, save = False):

	global binResults, binDates

	# listy = [i for i in listy if i != None]

	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("calcRunLength:")
	print()
	# print("listy:")
	# print(listy)

	defaultList = [0 for i in range(transitionCount)]
	defaultDates = ["-" for i in range(transitionCount)]

	binResults = defaultMatrix(len(quantiles) + 1, len(quantiles) + 1, defaultList)
	binDates = defaultMatrix(len(quantiles) + 1, len(quantiles) + 1, defaultDates)

	# print()
	# printer(binResults)
	# print()

	day = 0
	currentBin = -1
	prev = [-1, -1]
	currentCount = 1
	initDate = datetime.strptime(startingDate, "%Y%m%d")

	for i in range(len(listy)):
		# print(i)
		a = listy[i]

		if i == len(listy) - 1:
			b = -1

		else:
			b = listy[i + 1]

		if a == None or b == None:
			# print(f"a: {a}, b: {b} => CONTINUE")
			continue

		# print(f"{a} => {b}")
		# print("prev:", prev)

		if prev == [a, b]:
			# print("A: prev == [a, b]")
			currentCount += 1

		else:
			if prev != [-1, -1]:
				tempResults = binResults[prev[0]][prev[1]]
				# tempDates = binDates[prev[0]][prev[1]]

				for k in range(transitionCount):
					# print(f"k: {k} => {tempResults[k]}")

					if currentCount > tempResults[k]:
						binResults[prev[0]][prev[1]][k] = currentCount
						currentDate = datetime.strftime(initDate + timedelta(days = day), "%Y-%m-%d")
						binDates[prev[0]][prev[1]][k] = currentDate
						# print(f"B2 ({currentDate}): binResults[{prev[0]}][{prev[1]}][{k}] = {currentCount}")
						break

			# print("prev:", prev)
			prev = [a, b]
			# print("currentCount:", currentCount)
			# print(f"B1: prev = [{a}, {b}]")

			currentCount = 1

		# print()

		if (i + 1) % 24 == 0:
			day += 1

	printer(binResults)
	print()
	binPrinter(save = save, saveName = f"$ - {startingDate} to {endDate} - {len(quantiles)}Q, {transitionCount}TC, n = {number_of_cosines}")


def plotDay(date, csv = False, h1 = 0, h2 = 24, linear = False, poly = False, fourier = False,
						quantile = False, n = number_of_cosines, yList = [], endDate = None, hourly = True,
						scatter = False, q = 0.5, markov = False, s = 5, legend = True, xlabel = True,
						legendLabel = None, color = "#2F4F4F", save = True):

	print("plotDay n={}".format(n))
	print("endDate =>", endDate)

	if fourier:
		plt.figure("Net Demand")
		# print("Wind Power")

	else:
		plt.figure("Net Demand")

	title = "Net Demand"
	dList = []

	# xlabelText = f"{dataType} Wind Speed"
	xlabelText = "Time (hour)"

	if xlabel:
		plt.xlabel(xlabelText)

	plt.ylabel("net demand (MW)")

	if legend:
		label = date[0:4] + "-" + date[4:6] + "-" + date[6:8]

	else:
		label = "_nolegend_"

	if endDate:
		label += " to " + endDate[0:4] + "-" + endDate[4:6] + "-" + endDate[6:8]

	if legendLabel:
		label = legendLabel

	if len(yList) == 0:
		yList = extractIndex(getDayInfo(date), column, toFloat = True)

	# print("yList:")
	# print(yList[0:3])
	# print("--------------------------------")

	xList = list(range(len(yList)))
	xdata = np.asarray(xList, dtype = np.int32)
	ydata = np.asarray(yList, dtype = np.float32)

	if windPowerLimit:
		plt.gca().set_ylim(windPowerLimit)

	if scatter:
		alpha = 0.6
		if color:
			plt.scatter(xList, yList, label = label, alpha = alpha, s = s, color = color)

		else:
			plt.scatter(xList, yList, label = label, alpha = alpha, s = s)

	else:
		plt.plot(yList, label = label)

	if linear:
		linearReg(xdata, ydata)

	if poly:
		polyReg(xdata, ydata)

	if fourier:

		x, y = variables("x, y")

		model_dict = {y: fourier_series(x, f = w, mean = np.mean(ydata), n = n)}
		fit = Fit(model_dict, x = xdata, y = ydata)

		fit_result = fit.execute()
		fourierY = fit.model(x = xdata, **fit_result.params).y
		print(fit_result.params)

		diff = (ydata - fourierY) ** 2
		sse = sum(diff)

		title = "Net Demand, n={}, Turbines: {}, sse={}".format(n, number_of_turbines, round(sse, 2))
		plt.plot(xdata, fourierY, color = "purple", label = "Fourier")

	# plt.title(title)

	if quantile:
		# print("y:")
		# print(yList)
		quantileReg(xdata, ydata, date, endDate = endDate, q = q, markov = markov, zoom = True, hourly = hourly)

	if save:
		saveImage(getImageName(date, fourier, linear, poly, n = n, endDate = endDate, hourly = hourly,
													 quantile = quantile, q = q))

	if legend:
		# plt.legend()
		pass


def plotRange(startingDate, endDate, csv = False, fourier = False, linear = False,
							poly = False, n = number_of_cosines, hourly = True, quantile = False, q = 0.5,
							scatter = False, markov = False, s = 5, legendLabel = None,dataColor = "#2F4F4F"):

	print("plotRange n={}".format(n))

	label = startingDate[0:4] + "-" + startingDate[4:6] + "-" + startingDate[6:8]
	label += " to "
	label += endDate[0:4] + "-" + endDate[4:6] + "-" + endDate[6:8]

	yList = []

	days = getRangeDates(startingDate, endDate)

	for i in days:

		listy = getDayInfo(i)

		for j in listy:
			yList.append(float(j[column]))

	xList = list(range(0, len(yList)))
	print("length:", len(yList))
	# xdata = np.asarray(xList, dtype = np.int32)
	# ydata = np.asarray(yList, dtype = np.float32)

	plotDay(startingDate, poly = poly, linear = linear, fourier = fourier, n = n,
					yList = yList, endDate = endDate, hourly = hourly, quantile = quantile, q = q,
					scatter = scatter, save = False, markov = markov, s = s, legendLabel = legendLabel)

	saveImage(getImageName(startingDate, fourier = fourier, linear = linear, poly = poly, n = n,
												 endDate = endDate, plotType = "plotRange", quantile = quantile, q = q))


def plotTimeInterval(startingDate, endDate, csv = False, scatter = False, legend = False,
										 xlabel = True, plotType = "plotTimeInterval", save = True, color = None):

	listy = getRangeDates(startingDate, endDate)

	for i in listy:
		plotDay(i, csv = csv, save = False, scatter = scatter, legend = legend, xlabel = xlabel, color = color)

	if save:
		saveImage(getImageName(startingDate, endDate = endDate, plotType = plotType))



def plotGeoff(startingDate, endDate, hourly = True, n = number_of_cosines):

	yList = []

	days = getRangeDates(startingDate, endDate)

	for i in days:
		listy = getDayInfo(i, hourly = hourly)
		# yList = []

		for j in listy:

			temp = float(j[column])

			if temp >= 99:
				yList.append(np.nan)

			else:
				yList.append(float(j[column]))


	xList = list(range(0, len(yList)))
	xdata = np.asarray(xList, dtype = np.int32)
	ydata = np.asarray(yList, dtype = np.float32)

	x, y = variables("x, y")
	#w, = parameters("w")

	model_dict = {y: fourier_series(x, f = w, mean = np.mean(ydata), n = n)}

	fit = Fit(model_dict, x = xdata, y = ydata)
	fit_result = fit.execute()
	fourierY = fit.model(x = xdata, **fit_result.params).y
	print(fit_result.params)

	diff = (ydata - fourierY) ** 2
	sse = sum(diff)

	title = "Net Demand, n={}, sse={}".format(n, round(sse, 2))

	plotTimeInterval(startingDate, endDate, hourly = hourly, save = False)

	if hourly:
		plt.plot(xdata[0:24], fourierY[0:24], color = "black", label = "Fourier", ls = "--")

	else:
		plt.plot(xdata[0:144], fourierY[0:144], color = "black", label = "Fourier", ls = "--")

	# plt.title(title)
	# plt.legend()
	saveImage(getImageName(startingDate, endDate = endDate, plotType = "plotGeoff", n = n, hourly = hourly))


def plotGolbon(startingDate, endDate, legend = False, dataColor = None, scatter = True, plotLimit = "daily"):

	print("f:", f)
	# limit = int(1 / f)

	limit = 24

	if plotLimit == "annual":
		limit = 365

	print("limit: {}".format(limit))

	yList = getRangeDayInfo(startingDate, endDate)
	xList = list(range(len(yList)))

	xdata = np.asarray(xList, dtype = np.int32)
	ydata = np.asarray(yList, dtype = np.float32)

	print("--------------------------------------------")
	plotTimeInterval(startingDate, endDate, scatter = scatter, legend = False, xlabel = False, save = False, color = dataColor)
	quantileReg(xdata, ydata, startingDate, endDate = endDate, limit = limit)
	# plt.xlabel(f"{datify(startingDate)} to {datify(endDate)} {dataType}")
	plt.xlabel("Time of day (hour)\n"f"{datify(startingDate)} to {datify(endDate)}")
	
	handles, labels = plt.gca().get_legend_handles_labels()
	qLen = len(quantiles)
	legendOrder = [i for i in range(qLen, len(labels))] + [i for i in range(qLen)]

	dLen = len(labels) - qLen
	labels = labels[dLen:] + labels[:dLen]
	handles = handles[dLen:] + handles[:dLen]
	# plt.legend(handles, labels)
	saveImage(getImageName(startingDate, endDate = endDate, plotType = "plotGolbon"))


def plotAnnual(yearA, yearB, scatter = True, dataColor = None, qlw = 1):

	ans = []
	limit = 0

	for year in range(yearA, yearB + 1):

		print("currentYear: {}".format(year))
		startingDate = f"{year}0101"
		endDate = f"{year}1231"

		yList = getRangeDayInfo(startingDate, endDate)
		limit = len(yList)
		ans += yList

		plotDay(startingDate, endDate = endDate, yList = yList, scatter = True, legend = False,
						save = False, color = dataColor)

	xAns = list(range(len(ans)))
	xdata = np.asarray(xAns, dtype = np.int32)
	ydata = np.asarray(ans, dtype = np.float32)
	quantileReg(xdata, ydata, f"{yearA}0101", f"{yearB}1231", markov = True, limit = limit, qlw = qlw)

	# plt.title("Wind Power, n={}, Turbines: {}".format(number_of_cosines, number_of_turbines))
	plt.title("Wind Power for one turbine")
	plt.xlabel("Time (hour)\n "f"{yearA}-01-01 to {yearB}-12-31")
	# plt.xlabel(f"{yearA}-01-01 to {yearB}-12-31")
	plt.ylabel("Wind Power (MW)")
	saveImage(getImageName(f"{yearA}0101", fourier = True, n = number_of_cosines, endDate = f"{yearB}1231",
												 plotType = "plotAnnual", quantile = True))


def loopPlot(date, endDate = "", plotType = "plotDay", h1 = 0, h2 = 24,
						 linear = False, poly = False, fourier = False, hourly = True, n = 10,
						 scatter = False):

	for i in range(1, n + 1):

		print("-------- i={} --------".format(i))
		print()

		if plotType == "plotDay":
			plotDay(date, h1 = h1, h2 = h2, power = power, linear = linear, poly = poly, fourier = fourier, hourly = hourly, n = i, scatter = scatter)

		elif plotType == "plotRange":
			plotRange(date, endDate, power = power, linear = linear, poly = poly, fourier = fourier, hourly = hourly, n = i, scatter = scatter)

# -------------------------

# commands

# plotDay("20180204", quantile = True, scatter = True, q = 0.1, s = 10, markov = True)
plotRange("20060101", "20201231", quantile = True, scatter = True, markov = False, dataColor="#2F4F4F")

# plotRange("20210101", "20210130", quantile = True, scatter = True, markov = True)
# plotRange("20190101", "20190228", quantile = True, scatter = True, markov = True)
# plotTimeInterval("20060101", "20060112", scatter = False)
# plotGolbon("20190713", "20190822", scatter = True, plotLimit = "daily")
# plotGolbon("20180101", "20181231", legend = False, scatter = True,dataColor = "LightSeaGreen", plotLimit = "daily")
# plotGolbon("20180101", "20181231", scatter = True, plotLimit = "daily")
# plotRange("20200616", "20200816", scatter = True, quantile = True, markov = True)

# plotGolbon("20120101", "20120201", plotLimit = "annual")
# plotGolbon("20120101", "20120104", plotLimit = "daily")
# plotAnnual(2006, 2020, dataColor = "#2F4F4F", qlw = 2)


# plotGolbon("20180101", "20181231", scatter = True, plotLimit = "daily",dataColor = "#2F4F4F")

# plotDay("20200204", quantile = True, scatter = True, markov = True)
# plotRange("20180101", "20180101", quantile = True, scatter = True, markov = False)

# print("Wind Power (3):")
# print(calculateWindPower(3))


plt.show()
