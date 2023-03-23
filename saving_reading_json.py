import json
def export_json(file_name='btata.json',
	time_step=0.1,
	start_time=50,
	Gz_amp=0.4,
	Gz_time=15,
	Gph_max_amp=.7,
	Gph_time=4,
	Gx_amp = 0.4,
	Gx_time=24,
	no_of_rows =5,
	scaling_factor=2,
	noise_scaling=5,
	TR=2000,TE=80):
	line_items=[]
	myjson3 = {
	'time_step':time_step,
	'start_time':start_time,
	'Gz_amp':Gz_amp,
	'Gz_time':Gz_time,
	'Gph_max_amp':Gph_max_amp,
	'Gph_time':Gph_time,
	'Gx_amp' :Gx_amp,
	'Gx_time':Gx_time,
	'no_of_rows' :no_of_rows,
	'scaling_factor':scaling_factor,
	'noise_scaling':noise_scaling,
	'TR':TR,
	'TE':TE}
	line_items.append(myjson3)
	print(json.dumps(line_items,indent=4))
	with open(file_name,'w')as f:json.dump(myjson3,f,indent=4)
export_json()
with open('btata.json') as r:
	data = json.load(r)
print(data["Gph_max_amp"])