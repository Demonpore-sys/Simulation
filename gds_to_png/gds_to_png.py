import sys
import os.path
from gdsCAD import *
from PIL import Image, ImageDraw, ImageOps, ImageMath, ImageFont
import numpy as np

this_dir = os.path.dirname(__file__)

# l = core.GdsImport(os.path.abspath("Demonpore-Wafer-Map_0degree-Aligned_POREs-SLITs-actual-size_2020-05-27.GDS"), verbose=True)
# l2, l4 = utils.split_layers(a, [2,4])

BLACK_COLOR = 0
WHITE_COLOR = 1


class Simulation(object):
    def __init__(self, start_degree, end_degree, step_size=0.1, sweep_forward_then_backward=False,
                 image_output=True, csv_output=True, video_output_filename='video.mp4'):
        font_height = 20
        subset_font_height = 15
        font = ImageFont.truetype(os.path.abspath(os.path.join(this_dir, "FreeMono.ttf")), font_height)
        subset_font = ImageFont.truetype(os.path.abspath(os.path.join(this_dir, "FreeMono.ttf")), subset_font_height)

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
        print('adding slits to new image')
        l4_output = Image.new("1", (int_w + 1, int_h + 1), color=WHITE_COLOR)
        self.add_slits(l4_output, l4, quality_factor, int_min_x, int_min_y)
        l4_output.save('l4_slits.png')
        print('saved slits to new image')
        l4_mask = ImageOps.invert(l4_output.convert('L', dither=None)).convert('1', dither=None)
        l4_mask.save('l4_mask.png')
        print('saved slits mask image')
        print('adding pores to new image')
        l2_output = Image.new("1", (int_w + 1, int_h + 1), color=WHITE_COLOR)
        pore_points = self.add_pores(l2_output, l2, quality_factor, int_min_x, int_min_y)
        l2_output.save('l2_pores.png')
        print('saved pores to new image')
        l2_mask = ImageOps.invert(l2_output.convert('L', dither=None)).convert('1', dither=None)
        print('created pores mask image')
        print('pasting slits + pores into new image')
        combined = Image.new("1", (int_w + 1, int_h + 1), color=WHITE_COLOR)
        combined.paste(l2_output, (0, 0), l2_mask)
        combined.paste(l4_output, (0, 0), l4_mask)
        combined.save("l2_l4.png")
        print('saved pores+slits combination image')

        subsets = self.do_binning(pore_points)
        print('binned GDS points into dies/clusters')

        biggest_pore_group_dims = self.find_largest_pore_group_size(subsets)
        cropped_subset_height = biggest_pore_group_dims[0]
        cropped_subset_width = biggest_pore_group_dims[1]

        # rot, rot_x, rot_y, overlap_data = do_rotate(-2.27)
        # test = generate_cropped_image(rot, rot_x, rot_y, text='{} deg'.format(-2.27), overlap_data=overlap_data)

        images = []

        one_pixel_is_x_nanometers = l4_gds.unit/quality_factor
        if csv_output:
            csv_file = open('csv_output.csv', 'w')
            csv_file.write('rotation, pore group, number of pixels overlapping, number of meters per pixel, multiples pores overlapping\n')

        start_degree = int(start_degree * (1.0 / step_size))
        end_degree = int(end_degree * (1.0 / step_size))
        degree_list = range(start_degree, end_degree)
        if sweep_forward_then_backward:
            degree_list += range(start_degree * -1, (end_degree * -1) - 1, -1)
        for i in degree_list:
            i = i * step_size
            rot, rot_x, rot_y, overlap_data = self.do_rotate(i, l4_output, l2_output, l2_mask, subsets)
            # rot.save('rot_{}.png'.format(i))
            if csv_output:
                for col_i, col in enumerate(subsets):
                    for col_j, row in enumerate(col):
                        overlap_percentage, overlap_num, xs, ys = overlap_data[col_i][col_j]
                        pore_group = '{}_{}'.format(col_i, col_j)
                        csv_file.write('{}, {}, {}, {}, {}\n'.format(i, pore_group, overlap_num, one_pixel_is_x_nanometers, ''))
            if image_output:
                images.append(self.generate_cropped_image(rot, rot_x, rot_y,
                                                          cropped_subset_height, cropped_subset_width,
                                                          '{} deg'.format(i), font, font_height, subset_font,
                                                          overlap_data, subsets))
                images[-1].save('cropped{}.png'.format(i))
                print('just saved cropped{}.png'.format(i))

        if csv_output:
            csv_file.close()

        if image_output:
            images = [image.convert('RGB') for image in images]
            images[0].save('animation.gif',
                           save_all=True, append_images=images[1:], optimize=False, duration=15, loop=0)

            print('done with gif')
            import subprocess
            if os.path.isfile(video_output_filename):
                os.remove(video_output_filename)
            proc = subprocess.Popen(
                'ffmpeg -i animation.gif -movflags faststart -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" {}'
                .format(video_output_filename),
                shell=True)
            proc.communicate()
        print('done')

    @staticmethod
    def add_slits(image, l4, quality_factor, int_min_x, int_min_y):
        img1 = ImageDraw.Draw(image)
        # img1.rectangle(shape)#, fill ="# ffff33", outline ="red")
        for box in l4.elements:
            corner1, corner2 = (box.bounding_box * quality_factor) - (int_min_x, int_min_y)
            x1, y1 = corner1
            x2, y2 = corner2
            img1.rectangle(((x1, y1), (x2, y2)), fill=BLACK_COLOR, outline=0)

    @staticmethod
    def add_pores(image, l2, quality_factor, int_min_x, int_min_y):
        pore_points = []
        pore_point_cache = {}
        img1 = ImageDraw.Draw(image)
        for obj in l2.elements:
            corner1, corner2 = (obj.bounding_box * quality_factor) - (int_min_x, int_min_y)
            x1, y1 = map(int, corner1)
            x2, y2 = map(int, corner2)
            img1.rectangle(((x1, y1), (x2, y2)), fill=BLACK_COLOR, outline=0)

            # collect the points that make up the given rectangle
            w = int(abs(corner1[0] - corner2[0])) + 1
            h = int(abs(corner1[1] - corner2[1])) + 1
            xmin = int(min(corner1[0], corner2[0]))
            ymin = int(min(corner1[1], corner2[1]))
            if (w, h) not in pore_point_cache:
                pore_img = Image.new("1", (w, h), color=WHITE_COLOR)
                pore_draw = ImageDraw.Draw(pore_img)
                pore_draw.rectangle(((x1-xmin, y1-ymin), (x2-xmin, y2-ymin)), fill=BLACK_COLOR, outline=0)
                pore_pts = []
                for x in range(w):
                    for y in range(h):
                        if pore_img.getpixel((x, y)) == 0: # it's black
                            pore_pts.append((x, y))
                pore_point_cache[(w, h)] = pore_pts
            pore_pts = pore_point_cache[(w, h)]
            for coord in pore_pts:
                x, y = coord
                pore_points.append((x+xmin, y+ymin))
        return pore_points

    @staticmethod
    def do_binning(pore_points):
        x_bins = []
        final_bins = []
        points_sorted_by_x = sorted(pore_points, key=lambda x: x[0])
        points_sorted_by_y = sorted(pore_points, key=lambda x: x[1])
        print('sorted pore points for binning')
        o, bin_edges_x = np.histogram([x for x, y in points_sorted_by_x], bins=4)
        o, bin_edges_y = np.histogram([y for x, y in points_sorted_by_y], bins=4)
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

    @staticmethod
    def find_largest_pore_group_size(subsets):
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

    @staticmethod
    def generate_cropped_image(combined_image, x_off, y_off, sub_h, sub_w,
                               degree_rotation_text, degree_rotation_font, degree_rotation_font_height,
                               overlap_percent_font, overlap_data, subsets):
        per_group_text_padding = 75
        subset_image = Image.new("1", (sub_w * len(subsets),
                                       (sub_h + per_group_text_padding) * len(subsets[0])),
                                 color=WHITE_COLOR)
        draw = ImageDraw.Draw(subset_image)
        for i, col in enumerate(subsets):
            for j, row in enumerate(col):
                overlap_percentage, overlap_num, xs, ys = overlap_data[i][j]
                minx = min(xs)
                miny = min(ys)
                maxx = max(xs)
                maxy = max(ys)
                cropped_example = combined_image.crop((minx+x_off, miny+y_off, maxx+x_off, maxy+y_off))
                topleft_x = sub_w * i
                topleft_y = (sub_h+per_group_text_padding) * j
                subset_image.paste(cropped_example, (topleft_x, topleft_y))
                #draw.rectangle(((topleft_x, topleft_y), (topleft_x+sub_w, topleft_y+sub_h)), fill=None, outline=0)
                draw.text((topleft_x, topleft_y + sub_h), '{:.2f}%'.format(overlap_percentage), 0, font=overlap_percent_font)
        if degree_rotation_text:
            draw.text((0, subset_image.height-degree_rotation_font_height), degree_rotation_text, 0, font=degree_rotation_font)
        return subset_image

    @staticmethod
    def debug_calc(c_slits, c_pores):
        c_slits.show()
        c_pores.show()
        combined = Image.new("1", (c_slits.width, c_slits.height), color=WHITE_COLOR)
        combined.paste(c_slits, (0, 0))
        c_pores_mask = ImageOps.invert(c_pores.convert('L', dither=None)).convert('1', dither=None)
        combined.paste(c_pores, (0, 0), c_pores_mask)
        combined.show()
        pass

    @staticmethod
    def calc_overlap(slits, pores, pore_group_points, x_off=0, y_off=0):
        overlaps = []
        die_num = 0
        for i, col in enumerate(pore_group_points):
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
                print('percent_overlap {:.2f} pore group {} ({}_{})'.format(percent_overlap, die_num, i, j))
                overlaps[i].append((percent_overlap, len(num_overlapping), xs, ys))
                pass
        print('\n')
        return overlaps

    @staticmethod
    def do_rotate(degree, slit_image, pore_image, pore_image_mask, pore_group_points):
        rotated_slit_image = slit_image.rotate(degree, expand=False, fillcolor=WHITE_COLOR)
        rot_x = (rotated_slit_image.size[0]//2) - (pore_image.size[0]//2)
        rot_y = (rotated_slit_image.size[1]//2) - (pore_image.size[1]//2)
        #print('rot_x {} rot_y {}'.format(rot_x, rot_y))
        overlap_data = Simulation.calc_overlap(slits=rotated_slit_image, pores=pore_image,
                                               pore_group_points=pore_group_points, x_off=rot_x, y_off=rot_y)
        rotated_slit_image.paste(pore_image, (rot_x, rot_y), pore_image_mask)
        return rotated_slit_image, rot_x, rot_y, overlap_data


if __name__ == '__main__':
    #sim = Simulation(-2.6, 2.6, 0.01, sweep_forward_then_backward=True)
    # sim = Simulation(-0.5, 0.5, 0.1)
    sim = Simulation(-0.5, 0.5, 0.1, image_output=False, csv_output=True)
