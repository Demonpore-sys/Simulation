import sys
import csv
import os.path
from gdsCAD import *
from PIL import Image, ImageDraw, ImageOps, ImageMath, ImageFont
import numpy as np
import multiprocessing
this_dir = os.path.dirname(__file__)
# l = core.GdsImport(os.path.abspath("Demonpore-Wafer-Map_0degree-Aligned_POREs-SLITs-actual-size_2020-05-27.GDS"), verbose=True)
# l2, l4 = utils.split_layers(a, [2,4])

BLACK_COLOR = 0
WHITE_COLOR = 1


class Simulation(object):
    def __init__(self, start_degree, end_degree, step_size=0.1, sweep_forward_then_backward=False, quality=1, video_output_filename='video.mp4', use_multicore=False, save_images=True):
        self.font_height = 20
        self.subset_font_height = 15
        self.multicore_data = None

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
        quality_factor = quality # 1 (default)

        int_min_x = int(min(l4mins[0], l2mins[0])*quality_factor)
        int_min_y = int(min(l4mins[1], l2mins[1])*quality_factor)

        int_h = int(max(l2h, l4h)*quality_factor)
        int_w = int(max(l2w, l4w)*quality_factor)

        print('found bounding box for all pores on this die')
        print('adding slits to new image')
        l4_output = None
        l2_output = None
        if save_images:
            l4_output = Image.new("1", (int_w + 1, int_h + 1), color=WHITE_COLOR)
            l2_output = Image.new("1", (int_w + 1, int_h + 1), color=WHITE_COLOR)
        self.add_slits(l4_output, l4, quality_factor, int_min_x, int_min_y, save_images)
        print('adding pores to new image')
        pore_points = self.add_pores(l2_output, l2, quality_factor, int_min_x, int_min_y, save_images)
        subsets = self.do_binning(pore_points)
        print('binned GDS points into dies/clusters')
        if save_images:
            self.l4_output = l4_output
            self.l2_output = l2_output
            self.subsets = subsets
            self.sweep_forward_then_backward = sweep_forward_then_backward
            self.video_output_filename = video_output_filename
            self.start_degree = start_degree
            self.end_degree = end_degree
            self.step_size = step_size
            self.use_multicore = use_multicore
            self.int_w = int_w
            self.int_h = int_h
            self.do_simulation_with_images()

    def do_simulation_with_images(self):
        l4_output = self.l4_output
        l2_output = self.l2_output
        subsets = self.subsets
        font_height = self.font_height
        subset_font_height = self.subset_font_height
        int_w = self.int_w
        int_h = self.int_h
        start_degree = self.start_degree
        end_degree = self.end_degree
        step_size = self.step_size
        sweep_forward_then_backward = self.sweep_forward_then_backward
        video_output_filename = self.video_output_filename
        use_multicore = self.use_multicore

        l4_output.save('l4_slits.png')
        print('saved slits to new image')
        l4_mask = ImageOps.invert(l4_output.convert('L', dither=None)).convert('1', dither=None)
        l4_mask.save('l4_mask.png')
        print('saved slits mask image')
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



        biggest_pore_group_dims = self.find_largest_pore_group_size(subsets)
        cropped_subset_height = biggest_pore_group_dims[0]
        cropped_subset_width = biggest_pore_group_dims[1]

        # rot, rot_x, rot_y, overlap_data = do_rotate(-2.27)
        # test = generate_cropped_image(rot, rot_x, rot_y, text='{} deg'.format(-2.27), overlap_data=overlap_data)

        images = []

        start_degree = int(start_degree * (1.0 / step_size))
        end_degree = int(end_degree * (1.0 / step_size))
        degree_list = range(start_degree, end_degree)
        if sweep_forward_then_backward:
            degree_list += range(start_degree * -1, (end_degree * -1) - 1, -1)

        if not use_multicore:
            for i in degree_list:
                image, overlap_data = self.get_rotated_and_cropped_image(
                    i, step_size,
                    l4_output, l2_output, l2_mask, subsets,
                    cropped_subset_height, cropped_subset_width, font_height)
                images.append(image)

            self.save_images_to_animation(images, video_output_filename)
        else:
            self.multicore_data = (
                degree_list, step_size,
                l4_output, l2_output, l2_mask, subsets,
                cropped_subset_height, cropped_subset_width, font_height)


    @staticmethod
    def save_images_to_animation(images, video_output_filename):
        # if save_image:
        #     for i, degree_to_rotate, image in image_list:
        #         image.save('cropped{}.png'.format(degree_to_rotate))

        images = [image.convert('RGB') for image in images]
        images[0].save('animation.gif',
                       save_all=True, append_images=images[1:], optimize=False, duration=1, loop=0)

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
    def get_rotated_and_cropped_image(degree_to_rotate, step_size,
                                      slits_image, pores_image, pores_paste_mask, pore_pixel_locations,
                                      cropped_subset_height, cropped_subset_width, font_height):
        return_as_array = False
        if not isinstance(slits_image, Image.Image):
            slits_image = Simulation.convert_multiprocess_image(slits_image)
            pores_image = Simulation.convert_multiprocess_image(pores_image)
            pores_paste_mask = Simulation.convert_multiprocess_image(pores_paste_mask)
            return_as_array = True


        degree_to_rotate = degree_to_rotate * step_size
        rot, rot_x, rot_y, overlap_data = Simulation.do_rotate(degree_to_rotate, slits_image, pores_image, pores_paste_mask, pore_pixel_locations)
        print('about to generate cropped image')
        # rot.save('rot_{}.png'.format(i))
        image = Simulation.generate_cropped_image(rot, rot_x, rot_y,
                                            cropped_subset_height, cropped_subset_width,
                                            '{} deg'.format(degree_to_rotate), font_height,
                                            overlap_data, pore_pixel_locations)
        print('just saved cropped{}.png'.format(degree_to_rotate))
        if return_as_array:
            image = Simulation.convert_multiprocess_image(image)
        return (image, overlap_data)

    @staticmethod
    def convert_multiprocess_image(image_data, size=None, mode=None):#w=None, h=None, mode=None):
        if isinstance(image_data, Image.Image):
            return np.array(image_data)
            image_data = image_data.getdata()
            # w = image_data.width
            # h = image_data.height
            size = image_data.size
            mode = image_data.mode
            return (image_data, size, mode)
            #
        else: #it's raw data
            return Image.fromarray(image_data)
            n = Image.new(mode, size)#(w,h))
            n.putdata(image_data)
            return n
            #

    @staticmethod
    def add_slits(image, l4, quality_factor, int_min_x, int_min_y, save_images):
        if save_images:
            img1 = ImageDraw.Draw(image)
        slits_csv_file = open('slits_corner_coords.csv', 'w')
        slits_csv_writer = csv.writer(slits_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        slits_csv_writer.writerow(['x1', 'y1', 'x2', 'y2'])
        # img1.rectangle(shape)#, fill ="# ffff33", outline ="red")
        for box in l4.elements:
            corner1, corner2 = (box.bounding_box * quality_factor) - (int_min_x, int_min_y)
            x1, y1 = corner1
            x2, y2 = corner2
            if save_images:
                img1.rectangle(((x1, y1), (x2, y2)), fill=BLACK_COLOR, outline=0)
            slits_csv_writer.writerow((x1, y1, x2, y2))
        slits_csv_file.close()

    @staticmethod
    def add_pores(image, l2, quality_factor, int_min_x, int_min_y, save_images):
        pore_points = []
        pore_point_cache = {}
        if save_images:
            img1 = ImageDraw.Draw(image)
        pores_csv_file = open('pores_corner_coords.csv', 'w')
        pores_csv_writer = csv.writer(pores_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        pores_csv_writer.writerow(['x1', 'y1', 'x2', 'y2'])

        for obj in l2.elements:
            corner1, corner2 = (obj.bounding_box * quality_factor) - (int_min_x, int_min_y)
            x1, y1 = map(int, corner1)
            x2, y2 = map(int, corner2)
            if save_images:
                img1.rectangle(((x1, y1), (x2, y2)), fill=BLACK_COLOR, outline=0)
            else:
                pores_csv_writer.writerow((x1, y1, x2, y2))
                continue
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
        pores_csv_file.close()
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
                               degree_rotation_text, degree_rotation_font_height,
                               overlap_data, subsets):
        degree_rotation_font = ImageFont.truetype(os.path.abspath(os.path.join(this_dir, "FreeMono.ttf")), 15)
        overlap_percent_font = ImageFont.truetype(os.path.abspath(os.path.join(this_dir, "FreeMono.ttf")), 20)
        per_group_text_padding = 75
        subset_image = Image.new("1", (sub_w * len(subsets),
                                       (sub_h + per_group_text_padding) * len(subsets[0])),
                                 color=WHITE_COLOR)
        draw = ImageDraw.Draw(subset_image)
        for i, col in enumerate(subsets):
            for j, row in enumerate(col):
                overlap_percentage, xs, ys = overlap_data[i][j]
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
                print('percent_overlap {:.2f} die num {}'.format(percent_overlap, die_num))
                overlaps[i].append((percent_overlap, xs, ys))
                pass
        print('\n')
        return overlaps

    @staticmethod
    def do_rotate(degree, slit_image, pore_image, pore_image_mask, pore_group_points):
        rotated_slit_image = slit_image.rotate(degree, expand=False, fillcolor=WHITE_COLOR)
        rot_x = (rotated_slit_image.size[0]//2) - (pore_image.size[0]//2)
        rot_y = (rotated_slit_image.size[1]//2) - (pore_image.size[1]//2)
        print('rot_x {} rot_y {}'.format(rot_x, rot_y))
        overlap_data = Simulation.calc_overlap(slits=rotated_slit_image, pores=pore_image,
                                               pore_group_points=pore_group_points, x_off=rot_x, y_off=rot_y)
        rotated_slit_image.paste(pore_image, (rot_x, rot_y), pore_image_mask)
        return rotated_slit_image, rot_x, rot_y, overlap_data


# def really_do_multicore(send_q, rcv_q):
#     Simulation.unpack_for_multicore
def unpack_for_multicore(i, degree_to_rotate, step_size,
    slits_image, pores_image, pores_paste_mask, pore_pixel_locations,
    cropped_subset_height, cropped_subset_width, font_height):
    image, overlap = Simulation.get_rotated_and_cropped_image(
        degree_to_rotate, step_size,
        slits_image, pores_image, pores_paste_mask, pore_pixel_locations,
        cropped_subset_height, cropped_subset_width, font_height)
    #image_q.put((i, degree_to_rotate, image, overlap))
    return (i, degree_to_rotate, image, overlap)

def do_multicore(video_output_filename, args):
    # multiprocess this
    print('starting multicore procedure using {} cores'.format(multiprocessing.cpu_count()))
    (degree_list, step_size,
     slits_image, pores_image, pores_image_mask,
     subsets,
     cropped_subset_height, cropped_subset_width, font_height) = args
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    send_q = multiprocessing.Queue()
    rcv_q = multiprocessing.Queue()
    slits_as_array = Simulation.convert_multiprocess_image(slits_image)
    pores_as_array = Simulation.convert_multiprocess_image(pores_image)
    pores_mask_as_array = Simulation.convert_multiprocess_image(pores_image_mask)
    jobs = []
    num_degrees = len(degree_list)
    job_results = []
    print('starting to submit jobs')
    for i, degree_num in enumerate(degree_list):
        job = (i, degree_num, step_size,
               slits_as_array, pores_as_array, pores_mask_as_array,
               #None, None, None,
               subsets,
               cropped_subset_height, cropped_subset_width, font_height)
        #jobs.append((send_q, rcv_q))
        #send_q.put(job)
        print('submitted job {} of {}'.format(i, num_degrees))
        job_results.append(p.apply_async(unpack_for_multicore,
                                job))
    # result = p.map(Simulation.unpack_for_multicore,
    #                 jobs)

    print('closing the job pool to new jobs')
    # close the pool so new jobs can't be added
    p.close()
    print('waiting for jobs to finish')
    # wait for all current jobs to finish running
    p.join()
    print('all jobs finished')
    results = [jb.get() for jb in job_results]
    print('got all results')
    # (i, degree_to_rotate, image, overlap)
    result_data = []
    #for degree_num in degree_list:
    #    result_data.append(rcv_q.get(block=True, timeout=100))
    #results_sorted = sorted(result_data, key=lambda data: data[0])
    results_sorted = sorted(results, key=lambda data: data[0])
    images = [Simulation.convert_multiprocess_image(data[2]) for data in results_sorted]


    with open('overlap_multicore.csv', 'w') as overlap_csv:
        overlap_csv.write('degree, group, percentage\n')
        for data in results_sorted:
            deg = data[1]
            overlap_data = data[3]
            for i, col in enumerate(subsets):
                for j, row in enumerate(col):
                    overlap_percentage, xs, ys = overlap_data[i][j]
                    overlap_csv.write('{},{},{}\n'.format(deg, '{}_{}'.format(i,j), overlap_percentage))

    Simulation.save_images_to_animation(images, video_output_filename)
    pass


if __name__ == '__main__':
    # WARNING!!!! Don't use quality of more than 1 on a normal PC is you're using save_images=Treu  ... it will eat up all your RAM and lock up your machine
    sim = Simulation(-0.1, 0.1, 0.1, sweep_forward_then_backward=True, use_multicore=True, quality=1000, save_images=False)

    # sim = Simulation(-0.1, 0.1, 0.1, sweep_forward_then_backward=True, use_multicore=True, quality=1)
    if sim.multicore_data:
        do_multicore('multicore_made.mp4', sim.multicore_data)
