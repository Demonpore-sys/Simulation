import os.path
from gdsCAD import *
from PIL import Image, ImageDraw


#l = core.GdsImport(os.path.abspath("Demonpore-Wafer-Map_0degree-Aligned_POREs-SLITs-actual-size_2020-05-27.GDS"), verbose=True)
#l2, l4 = utils.split_layers(a, [2,4])

l4_gds = core.GdsImport(os.path.abspath("die5_from_topleft_layer4_slits_shown.GDS"), verbose=True)
l2_gds = core.GdsImport(os.path.abspath("die5_from_topleft_layer2_shown.GDS"), verbose=True)

print('imported GDS units {} precision {}'.format(l4_gds.unit, l4_gds.precision))
tl_4 = l4_gds.top_level()
l4=tl_4[0]

tl_2 = l2_gds.top_level()
l2=tl_2[0]

l4mins, l4maxs = l4.bounding_box
l4w = l4maxs[0] - l4mins[0]
l4h = l4maxs[1] - l4mins[1]

l2mins, l2maxs = l2.bounding_box
l2w = l2maxs[0] - l2mins[0]
l2h = l2maxs[1] - l2mins[1]

int_min_x = int(min(l4mins[0], l2mins[0]))
int_min_y = int(min(l4mins[1], l2mins[1]))

int_h = int(max(l2h, l4h))
int_w = int(max(l2w, l4w))


def add_slits(image):
	img1 = ImageDraw.Draw(image)
	# img1.rectangle(shape)#, fill ="# ffff33", outline ="red")
	for box in l4.elements:
		corner1, corner2 = box.bounding_box-(int_min_x, int_min_y)
		x1, y1 = corner1
		x2, y2 = corner2
		img1.rectangle(((x1, y1), (x2, y2)), fill=0, outline=0)


def add_pores(image):
	for obj in l2.elements:
		for coord in obj.points:
			x, y = coord
			xx = int(x - int_min_x)
			yy = int(y - int_min_y)
			try:
				image.putpixel((xx, yy), 0)
			except IndexError as e:
				print('{} {} w {} h {}'.format(xx, yy, int_w, int_h))
				raise e


l4_output = Image.new("1", (int_w+1, int_h+1), color=1)
add_slits(l4_output)
l4_output.save('l4_slits.png')

l2_output = Image.new("1", (int_w + 1, int_h + 1), color=1)
add_pores(l2_output)
l2_output.save('l2_pores.png')

combined = Image.new("1", (int_w + 1, int_h + 1), color=1)
add_slits(combined)
add_pores(combined)
combined.save("l2_l4.png")

print('done')
