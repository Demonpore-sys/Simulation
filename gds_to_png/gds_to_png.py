import sys
import os.path
from gdsCAD import *
from PIL import Image, ImageDraw, ImageOps, ImageMath, ImageFont
import numpy as np

# l = core.GdsImport(os.path.abspath("Demonpore-Wafer-Map_0degree-Aligned_POREs-SLITs-actual-size_2020-05-27.GDS"), verbose=True)
# l2, l4 = utils.split_layers(a, [2,4])

l4_gds = core.GdsImport(os.path.abspath("die5_from_topleft_layer4_slits_shown.GDS"), verbose=True)
l2_gds = core.GdsImport(os.path.abspath("die5_from_topleft_layer2_shown.GDS"), verbose=True)

print('imported GDS units {} precision {}'.format(l4_gds.unit, l4_gds.precision))
tl_4 = l4_gds.top_level()
l4 = tl_4[0]

tl_2 = l2_gds.top_level()
l2 = tl_2[0]

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
        corner1, corner2 = box.bounding_box - (int_min_x, int_min_y)
        x1, y1 = corner1
        x2, y2 = corner2
        img1.rectangle(((x1, y1), (x2, y2)), fill=0, outline=0)

pore_group_bounds = []
pore_points_x = []
pore_points_y = []
pore_points = []


def add_pores(image, off_x=0, off_y=0, collect_points=False):
    for obj in l2.elements:
        for coord in obj.points:
            x, y = coord
            xx = int(x - int_min_x)
            yy = int(y - int_min_y)
            try:
                image.putpixel((xx+off_x, yy+off_y), 0)
                if collect_points:
                    pore_points_x.append(xx)
                    pore_points_y.append(yy)
                    pore_points.append((xx, yy))
            except IndexError as e:
                print('{} {} w {} h {}'.format(xx, yy, int_w, int_h))
                raise e


l4_output = Image.new("1", (int_w + 1, int_h + 1), color=1)
add_slits(l4_output)
l4_output.save('l4_slits.png')
#l4_mask = ImageMath.eval('1-(a)', a=l4_output).convert('1', dither=None)
#l4_mask = 1 - numpy.asarray(l4_output)
l4_mask = ImageOps.invert(l4_output.convert('L', dither=None)).convert('1', dither=None)
l4_mask.save('l4_mask.png')

l2_output = Image.new("1", (int_w + 1, int_h + 1), color=1)
add_pores(l2_output, collect_points=True)
l2_output.save('l2_pores.png')
l2_mask = ImageOps.invert(l2_output.convert('L', dither=None)).convert('1', dither=None)


combined = Image.new("1", (int_w + 1, int_h + 1), color=1)
combined.paste(l2_output, (0,0), l2_mask)
combined.paste(l4_output, (0,0), l4_mask)
combined.save("l2_l4.png")


points_sorted_by_x = sorted(pore_points, key=lambda x: x[0])
points_sorted_by_y = sorted(pore_points, key=lambda x: x[1])
def do_binning():
    x_bins = []
    y_bins = []
    o, bin_edges_x = np.histogram([x for x, y in points_sorted_by_x], bins=4)
    o, bin_edges_y = np.histogram([y for x, y in points_sorted_by_y], bins=4)
    #occurence, bin_edges = np.histogram([1, 2, 3], bins=16)
    for bin in range(4):
        x_bins.append([])
        x_bins[-1] = [point for point in points_sorted_by_x if point[0] >= bin_edges_x[bin] and point[0] <= bin_edges_x[bin + 1]]
    for col in x_bins:
        y_bins.append([])
        for bin in range(4):
            y_bins[-1].append([])
            y_bins[-1][-1] = [point for point in col if point[1] >= bin_edges_y[bin] and point[1] <= bin_edges_y[bin + 1]]
    return y_bins

subsets = do_binning()


def find_largest_pore_group_size():
    biggest_pore_group_dims = [0, 0]
    for i, col in enumerate(subsets):
        for j, row in enumerate(col):
            xs = [x for x, y in row]
            ys = [y for x, y in row]
            minx = min(xs)
            miny = min(ys)
            maxx = max(xs)
            maxy = max(ys)
            biggest_pore_group_dims = (max(biggest_pore_group_dims[0], maxx - minx),
                                       max(biggest_pore_group_dims[1], maxy - miny))
    biggest_pore_group_dims = (biggest_pore_group_dims[0]+20, biggest_pore_group_dims[1]+20)
    return biggest_pore_group_dims

biggest_pore_group_dims = find_largest_pore_group_size()

font_height = 15
font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", font_height)


def generate_cropped_image(combined_image, x_off=0, y_off=0, text=''):
    subset_image = Image.new("1", (biggest_pore_group_dims[0] * len(subsets),
                                   biggest_pore_group_dims[1] * len(subsets[0])), color=1)
    for i, col in enumerate(subsets):
        for j, row in enumerate(col):
            xs = [x for x, y in row]
            ys = [y for x, y in row]
            minx = min(xs)
            miny = min(ys)
            maxx = max(xs)
            maxy = max(ys)
            cropped_example = combined_image.crop((minx+x_off, miny+y_off, maxx+x_off, maxy+y_off))
            topleft_x = biggest_pore_group_dims[0]*i
            topleft_y = biggest_pore_group_dims[1]*j
            subset_image.paste(cropped_example, (topleft_x, topleft_y))
    if text:
        draw = ImageDraw.Draw(subset_image)
        draw.text((0, subset_image.height-font_height), text, 0, font=font)
    return subset_image

images = [generate_cropped_image(combined, text='0 deg')]
images[0].save('cropped0.png')

def do_rotate(degree):
    ret = l4_output.rotate(degree, expand=True, fillcolor=1)
    rot_x = (ret.size[0]//2) - (l2_output.size[0]//2)
    rot_y = (ret.size[1]//2) - (l2_output.size[1]//2)
    # l2_resized = Image.new("1", (ret.width, ret.height), color=1)
    # l2_resized.paste(l2_output, (rot_x, rot_y))# ImageOps.invert(l2_output.convert('L')).convert('1'))
    # l2_resized.save('l2_resized.png')
    # l2_mask = ImageOps.invert(l2_resized.convert('L', dither=None)).convert('1', dither=None)
    # ret.paste(l2_resized, (0,0), l2_mask)
    #ret.paste(l2_output, (rot_x, rot_y), l2_mask)
    add_pores(ret, rot_x, rot_y)
    return ret, rot_x, rot_y


# rot, rot_x, rot_y = do_rotate(45)
# rot.save('l4_rot_add_l2.png')
for i in range(1, 30):
    i=i / 10.
    rot, rot_x, rot_y = do_rotate(i)
    images.append(generate_cropped_image(rot, rot_x, rot_y, text='{} deg'.format(i)))
    images[-1].save('cropped{}.png'.format(i))
    print('just saved cropped{}.png'.format(i))

images = [image.convert('RGB') for image in images]
images[0].save('animation.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=15, loop=0)

print('done')
