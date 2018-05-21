#zeverdinge importe hier
import sys
import numpy
import plotly
import plotly.graph_objs as go
import math




#maken van twee arrays, carray en sarray
sarray = []
carray = []
darray = []
datarray = [".5.txt",".50.txt",".500.txt",".5000.txt",".50000.txt"]

waarden = [5,50,500,5000,50000]


def gemrekenen(naamding):
	i = 0
	sarray = []
	carray = []
	darray = []
	fileding = open(str(naamding),"r")
	#lusje om alle lijntjes af te lopen
	for line in fileding:
		i += 1
		if (i >= 0):
			#print(line[-11:-2])
			floatding = float(str(line[-11:-2]))
			if (len(line) == 0):
				carray.append(0)
				sarray.append(0)
				darray.append(0)
				print("amai!")
				break
			if (line[0] == 'c'):
				carray.append(floatding)
			if (line[0] == 's'):
				sarray.append(floatding)
			if (line[0] == 'd'):
				darray.append(floatding)
		if (i == 350):
			break

	#gemiddeldes uitrekenen
	cemiddelde =  (numpy.mean(carray))
	semiddelde =  (numpy.mean(sarray))
	demiddelde =  (numpy.mean(darray))
	semiddelde =  numpy.round(semiddelde,7)
	cemiddelde =  numpy.round(cemiddelde,7)
	demiddelde =  numpy.round(demiddelde,7)
	gemwaarden = [cemiddelde,semiddelde,demiddelde]
	print(naamding+":")
	print(gemwaarden)
	return gemwaarden


def speciaalgemrekenen(naamding):
	i = 0
	fileding = open(str(naamding),"r")
	#lusje om alle lijntjes af te lopen
	for line in fileding:
		i += 1
		if (i > 1):
			#print(line[-11:-2])
			floatding = float(str(line[-11:-2]))
			sarray.append(floatding)
		if (i == 101):
			break

	#gemiddeldes uitrekenen
	semiddelde =  (numpy.mean(sarray))
	semiddelde =  numpy.round(semiddelde,7)
	gemwaarden = [0,semiddelde,0]
	print(naamding+":")
	print(gemwaarden)
	return gemwaarden




if __name__ == "__main__":
	naam = "n-body"
	cemwaarden = []
	semwaarden = []
	demwaarden = []
	for i in range(5):
            cemwaarden = []
            semwaarden = []
            demwaarden = []
            for j in range(6):
                #dees is echt een van de dwaaste manieren om dees te doen denkik
                #maar het werkt wel dus boeien lmao 
                tekstfile = "n-body" + str(j) + datarray[i]
                print(tekstfile)
                if (i == 0):
                    gemwaarde = speciaalgemrekenen(tekstfile)
                else:
                    gemwaarde = gemrekenen(tekstfile)

                cemwaarden.append(gemwaarde[0])
                if (math.isnan(gemwaarde[1])):
                    semwaarden.append(0)
                else:
                    semwaarden.append(gemwaarde[1])

                demwaarden.append(gemwaarde[2])
            cemdata = [go.Bar(x = [0,1,2,3,4,5], y = cemwaarden, name = "Computation time: bodies"+str(i))]
            semdata = [go.Bar(x = [0,1,2,3,4,5], y = semwaarden, name = "Simulation time: bodies"+str(i))]
            demdata = [go.Bar(x = [0,1,2,3,4,5], y = demwaarden, name = "Transfer time: bodies"+str(i))]
            cemout = {'title': "Computation time, "+str(waarden[i])+" bodies"}
            semout = {'title': "Simulation time, "+str(waarden[i])+" bodies"}
            demout = {'title': "Transfer time, "+str(waarden[i])+" bodies"}
            plotly.offline.plot({'data':cemdata,'layout':cemout},filename='hist_computation'+str(waarden[i])+'.html',image='png', image_filename='hist_computation'+str(waarden[i]))
            plotly.offline.plot({'data':semdata,'layout':semout},filename='hist_simulation'+str(waarden[i])+'.html',image='png', image_filename='hist_simulation'+str(waarden[i]))
            plotly.offline.plot({'data':demdata,'layout':demout},filename='hist_transfer'+str(waarden[i])+'.html',image='png', image_filename='hist_transfer'+str(waarden[i]))





