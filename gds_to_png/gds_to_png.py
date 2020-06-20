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

# how many decimal places to keep... 1 means none
quality_factor = 1

int_min_x = int(min(l4mins[0], l2mins[0])*quality_factor)
int_min_y = int(min(l4mins[1], l2mins[1])*quality_factor)

int_h = int(max(l2h, l4h)*quality_factor)
int_w = int(max(l2w, l4w)*quality_factor)

print('found bounding box for all pores on this die')


def add_slits(image):
    img1 = ImageDraw.Draw(image)
    # img1.rectangle(shape)#, fill ="# ffff33", outline ="red")
    for box in l4.elements:
        corner1, corner2 = box.bounding_box - (int_min_x, int_min_y)
        x1, y1 = corner1
        x2, y2 = corner2
        img1.rectangle(((x1, y1), (x2, y2)), fill=0, outline=0)

# pore_points_x = []
# pore_points_y = []
pore_points = []


def add_pores(image, off_x=0, off_y=0, collect_points=False):
    for obj in l2.elements:
        for coord in obj.points:
            x, y = coord
            xx = int((x*quality_factor) - int_min_x)
            yy = int((y*quality_factor) - int_min_y)
            try:
                image.putpixel((xx+off_x, yy+off_y), 0)
                if collect_points:
                    # pore_points_x.append(xx)
                    # pore_points_y.append(yy)
                    pore_points.append((xx, yy))
            except IndexError as e:
                print('{} {} w {} h {}'.format(xx, yy, int_w, int_h))
                raise e


print('adding slits to new image')
l4_output = Image.new("1", (int_w + 1, int_h + 1), color=1)
add_slits(l4_output)
l4_output.save('l4_slits.png')
print('saved slits to new image')
#l4_mask = ImageMath.eval('1-(a)', a=l4_output).convert('1', dither=None)
#l4_mask = 1 - numpy.asarray(l4_output)
l4_mask = ImageOps.invert(l4_output.convert('L', dither=None)).convert('1', dither=None)
l4_mask.save('l4_mask.png')
print('saved slits mask image')

print('adding pores to new image')
l2_output = Image.new("1", (int_w + 1, int_h + 1), color=1)
add_pores(l2_output, collect_points=True)
l2_output.save('l2_pores.png')
print('saved pores to new image')
l2_mask = ImageOps.invert(l2_output.convert('L', dither=None)).convert('1', dither=None)
print('created pores mask image')
print('pasting slits + pores into new image')
combined = Image.new("1", (int_w + 1, int_h + 1), color=1)
combined.paste(l2_output, (0, 0), l2_mask)
combined.paste(l4_output, (0, 0), l4_mask)
combined.save("l2_l4.png")
print('saved pores+slits combination image')

points_sorted_by_x = sorted(pore_points, key=lambda x: x[0])
points_sorted_by_y = sorted(pore_points, key=lambda x: x[1])
print('sorted pore points for binning')


def do_binning():
    x_bins = []
    final_bins = []
    o, bin_edges_x = np.histogram([x for x, y in points_sorted_by_x], bins=4)
    o, bin_edges_y = np.histogram([y for x, y in points_sorted_by_y], bins=4)
    #occurence, bin_edges = np.histogram([1, 2, 3], bins=16)
    for bin in range(4):
        x_bins.append([])
        x_bins[-1] = [point for point in points_sorted_by_x if point[0] >= bin_edges_x[bin] and point[0] <= bin_edges_x[bin + 1]]
    for col in x_bins:
        y_bins = []
        for row_num in range(4):
            group_bin = [point for point in col if point[1] >= bin_edges_y[row_num] and point[1] <= bin_edges_y[row_num + 1]]
            y_bins.append(group_bin)
        final_bins.append(y_bins)
    return final_bins


subsets = do_binning()
print('binned GDS points into dies/clusters')


def find_largest_pore_group_size():
    biggest_pore_group_dims = [0, 0]
    for i, col in enumerate(subsets):
        for j, row in enumerate(col):
            xs = [x for x, y in row]
            ys = [y for x, y in row]
            minx = min(xs)
            maxx = max(xs)
            miny = min(ys)
            maxy = max(ys)
            w = maxx - minx
            h = maxy - miny
            biggest_pore_group_dims = (max(biggest_pore_group_dims[0], w),
                                       max(biggest_pore_group_dims[1], h))
    biggest_pore_group_dims = (biggest_pore_group_dims[0]+20, biggest_pore_group_dims[1]+20)
    return biggest_pore_group_dims

biggest_pore_group_dims = find_largest_pore_group_size()

font_height = 20
subset_font_height = 15
font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", font_height)
subset_font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", subset_font_height)


def generate_cropped_image(combined_image, x_off=0, y_off=0, text='', overlap_data=None):
    per_group_text_padding = 75
    sub_h = biggest_pore_group_dims[0]
    sub_w = biggest_pore_group_dims[1]
    subset_image = Image.new("1", (sub_w * len(subsets),
                                   (sub_h + per_group_text_padding) * len(subsets[0])),
                             color=1)
    draw = ImageDraw.Draw(subset_image)
    for i, col in enumerate(subsets):
        for j, row in enumerate(col):
            overlap_percentage, xs, ys = overlap_data[i][j]
            # xs = [x for x, y in row]
            # ys = [y for x, y in row]
            minx = min(xs)
            miny = min(ys)
            maxx = max(xs)
            maxy = max(ys)
            cropped_example = combined_image.crop((minx+x_off, miny+y_off, maxx+x_off, maxy+y_off))
            topleft_x = sub_w * i
            topleft_y = (sub_h+per_group_text_padding) * j
            # if j:
            #     topleft_y+=
            subset_image.paste(cropped_example, (topleft_x, topleft_y))
            draw.rectangle(((topleft_x, topleft_y), (topleft_x+sub_w, topleft_y+sub_h)), fill=None, outline=0)
            draw.text((topleft_x, topleft_y + sub_h), '{:.2f}%'.format(overlap_percentage), 0, font=subset_font)
    if text:
        draw.text((0, subset_image.height-font_height), text, 0, font=font)
    return subset_image


def debug_calc(c_slits, c_pores):
    c_slits.show()
    c_pores.show()
    combined = Image.new("1", (c_slits.width, c_slits.height), color=1)
    combined.paste(c_slits, (0, 0))
    c_pores_mask = ImageOps.invert(c_pores.convert('L', dither=None)).convert('1', dither=None)
    combined.paste(c_pores, (0, 0), c_pores_mask)
    combined.show()
    pass


def calc_overlap(slits, pores, x_off=0, y_off=0):
    overlaps = []
    die_num = 0
    for i, col in enumerate(subsets):
        overlaps.append([])
        for j, row in enumerate(col):
            die_num += 1
            xs = [x for x, y in row]
            ys = [y for x, y in row]
            minx = min(xs)
            miny = min(ys)
            maxx = max(xs)
            maxy = max(ys)
            cropped_slit = slits.crop((minx + x_off, miny + y_off, maxx + x_off, maxy + y_off))
            cropped_pores = pores.crop((minx, miny, maxx, maxy))
            l = list(cropped_slit.getdata())
            ll = list(cropped_pores.getdata())
            num_pore_pix = [True for point in ll if point==0] # 0 is a dark pixel
            num_overlapping = [True for a, b in zip(l, ll) if a == b and a == 0]
            percent_overlap = (float(len(num_overlapping))/len(num_pore_pix))*100.
            #cropped_slit_np = np.asarray(cropped_slit)
            #cropped_pores_np = np.asarray(cropped_pores)
            #overlap_np = cropped_slit_np != cropped_pores_np
            #percent_overlap = 0.0
            #if np.sum(cropped_pores_np):
            #    percent_overlap = (np.sum(overlap_np) / float(np.sum(cropped_pores_np))) * 100
            #else:
            #    print('no pores??? seems like a bug')
            print('percent_overlap {:.2f} die num {}'.format(percent_overlap, die_num))
            overlaps[i].append((percent_overlap, xs, ys))
            #Image.fromarray
            pass
    print('\n')
    return overlaps


def do_rotate(degree):
    ret = l4_output.rotate(degree, expand=False, fillcolor=1)
    rot_x = (ret.size[0]//2) - (l2_output.size[0]//2)
    rot_y = (ret.size[1]//2) - (l2_output.size[1]//2)
    print('rot_x {} rot_y {}'.format(rot_x, rot_y))
    overlap_data = calc_overlap(slits=ret, pores=l2_output, x_off=rot_x, y_off=rot_y)
    # l2_resized = Image.new("1", (ret.width, ret.height), color=1)
    # l2_resized.paste(l2_output, (rot_x, rot_y))# ImageOps.invert(l2_output.convert('L')).convert('1'))
    # l2_resized.save('l2_resized.png')
    # ret.paste(l2_resized, (0,0), l2_mask)
    ret.paste(l2_output, (rot_x, rot_y), l2_mask)
    #add_pores(ret, rot_x, rot_y)
    return ret, rot_x, rot_y, overlap_data

rot, rot_x, rot_y, overlap_data = do_rotate(-2.27)
test = generate_cropped_image(rot, rot_x, rot_y, text='{} deg'.format(-2.27), overlap_data=overlap_data)

images = []
plus_minu_degree_range = 260
degree_list = range(-plus_minu_degree_range, plus_minu_degree_range) + range(plus_minu_degree_range, -plus_minu_degree_range-1, -1)
for i in degree_list:
    i=i / 100.
    rot, rot_x, rot_y, overlap_data = do_rotate(i)
    # rot.save('rot_{}.png'.format(i))
    images.append(generate_cropped_image(rot, rot_x, rot_y, text='{} deg'.format(i), overlap_data=overlap_data))
    images[-1].save('cropped{}.png'.format(i))
    draw = ImageDraw.Draw(rot)
    draw.rectangle(((rot_x, rot_y), (rot_x + l2_output.width, rot_y + l2_output.height)), fill=None, outline=0)
    rot.save('full{}.png'.format(i))
    print('just saved cropped{}.png'.format(i))

images = [image.convert('RGB') for image in images]
images[0].save('animation.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=15, loop=0)

print('done with gif')
import subprocess
subprocess.Popen('ffmpeg -i animation.gif -movflags faststart -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" video1.mp4', shell=True)
print('done')
