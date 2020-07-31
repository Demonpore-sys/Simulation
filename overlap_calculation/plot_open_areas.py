import openpyxl
from matplotlib import pyplot as plt
from matplotlib import gridspec

workbook = openpyxl.load_workbook("output_data_overlaps_30thousand1_xshifts.xlsx")
wsheet = workbook.active

x_shifts = []
# read the values of x shifts
for i in range(8,30009):
	x_shifts.append( float(wsheet["A"+str(i)].value) ) 

# pairs of columns in the Excel worksheet
columns = [['B','C'],['D','E'],['F','G'],['H','I'],['J','K'],['L','M'],['N','O'],['P','Q'],['R','S'],['T','U'],['V','W'],['X','Y'],['Z','AA'],['AB','AC'],['AD','AE'],['AF','AG']] 

for groupnum, cols in enumerate(columns):

	open_areas = []
	open_pore_counts = []

	for i in range(8,30009):
		open_areas.append( float(wsheet[cols[0]+str(i)].value) ) 
		open_pore_counts.append( float(wsheet[cols[1]+str(i)].value) ) 

	fig = plt.figure()
	fig.suptitle('Group '+str(groupnum+1)) 
	spec = gridspec.GridSpec(ncols=1, nrows=2,
	                         height_ratios=[4, 1])

	ax1 = fig.add_subplot(spec[0])
	ax1.plot(x_shifts, open_areas)
	ax1.set_ylabel('open pore area (nm$^2$)')

	ax2 = fig.add_subplot(spec[1])
	ax2.plot(x_shifts, open_pore_counts)
	ax2.set_xlabel('x axis shift from center of wafer (nm)')
	ax2.set_ylabel('num. open pores')

plt.show()
