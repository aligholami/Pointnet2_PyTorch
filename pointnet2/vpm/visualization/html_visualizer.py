import argparse
import os
import json
import shutil

CSS_BASE = "title { border-bottom: 3px solid #cc9900; color: #996600; font-size: 30px; } table, th, td { border: 2px solid black; border-collapse: collapse; padding: 5px; font-size: 15px; /* text-align: center; */ width: auto; } body {font-family: 'Roboto', sans-serif;}"
HTML_BASE = "<html> <style> @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300&display=swap'); {0} </style> <head> <title> {1} </title> </head> <body> <div> {2} </div> </body> </html> "
WEBSERVER_ROOT = 'http://aspis.cmpt.sfu.ca/projects/scan2cap'
DUMP_DIR = 'html_dumps'

class Scan2CapHTMLVisualizer:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def create_html_index(self, exp_names, dump_keys, matching_methods, matching_dump_keys):
    
        out = "<h1> Scan2Cap 2D Experiments </h1> <br> <table> <tr> <td> <strong> Train/Val Preds/GTs </strong> </td> <td> <strong> Experiment </strong>  </td> <td> <strong> BLEU_4 </strong></td> <td> <strong> CiDEr </strong> </td> <td> <strong> METEOR </strong> </td><td> <strong> ROUGE-L </strong> </td></tr>"
        dpath = self.args.index_file

        for exp in exp_names:
            f_paths = "<ul>"
            for k in dump_keys:
                f_paths += "<li> <a href='{0}'>{1}</a> </li> ".format(os.path.join(WEBSERVER_ROOT, DUMP_DIR, exp, k + '.html'), k.strip('pred_yo') + '.html')
            f_paths += "</ul>"
            info = "<p> {} </p>".format(exp)
            best_list = open(os.path.join(self.args.output_path, exp, 'best.txt'), 'r').readlines()

            # following indexes are based on the best.txt
            scores = {
                "bleu_4": "<strong> {0} </strong>".format(round(float(best_list[6].split(': ')[1]) * 100, 2)),
                "cider": "<strong> {0} </strong>".format(round(float(best_list[7].split(': ')[1]) * 100, 2)),
                "rouge": "<strong> {0} </strong>".format(round(float(best_list[8].split(': ')[1]) * 100, 2)),
                "meteor": "<strong> {0} </strong>".format(round(float(best_list[9].split(': ')[1]) * 100, 2))
            }

            out += "<tr> <td> {0} </td> <td> {1} </td> <td> {2} </td> <td> {3} </td> <td> {4} </td> <td> {5} </td></tr>".format(f_paths, info, scores["bleu_4"], scores["cider"], scores["rouge"], scores["meteor"])

        out += "</table>"
        
        out += "<br/>"

        out += "<h1> Frame Matching Methods </h1> <br> <table> <tr> <td> <strong> Ratio Bins </strong> </td> <td> <strong> Algorithm </strong> </td> <td> <strong> Average Pixel Ratio  </strong> </td><td> <strong> Accuracy </strong> </td> <td> <strong> Ratio Density </strong> </td><td> <strong> Ratio Per Object </strong> </td> </tr>"
        for method in matching_methods:
            f_paths = "<ul>"
            for k in matching_dump_keys:
                f_paths += "<li> <a href='{0}'>{1}</a> </li> ".format(os.path.join(WEBSERVER_ROOT, DUMP_DIR, method, k + '.html'), k.strip('pred_yo') + '.html')
            f_paths += "</ul>"
            info = "<p> {} </p>".format(method)
            avgs = json.load(open(os.path.join(self.args.matching_stats, method + '_avgs.json'), 'r'))
            out += "<tr> <td> {0} </td> <td> {1} </td> <td> {2} </td> <td> {3} </td>".format(f_paths, info, avgs['pixel_ratio'], avgs['matching_accuracy'])
            out += "<td width=500px> <img src='{0}' loading='lazy' style='width:500px;height:500px;' </td>".format(os.path.join(WEBSERVER_ROOT, self.args.plots_symlink, method, 'frame_density' + '.png'))
            out += "<td width=500px> <img src='{0}' loading='lazy' style='width:500px;height:500px;' </td></tr>".format(os.path.join(WEBSERVER_ROOT, self.args.plots_symlink, method, 'ratio_per_object' + '.png'))

        out += '</table>'
        if self.args.debug:
            print("Adding debug visualizations to the index.")
            out += "<br> <h1> Debug Visualizations </h1> <br> <table> <tr> <td> <strong> scene_id </strong> </td> </tr> "
            # scl_1 = list(set([item['scene_id'] for item in json.load(open(os.path.join(args.matches_path, 'debug_matches_train.json')))]))
            scl_2 = list(set([item['scene_id'] for item in json.load(open(os.path.join(args.matches_path, 'debug_matches_val.json')))]))
            scl = scl_2
            fpaths = "<ul>"
            for scene_id in scl:
                fpaths += "<li> <a href='{0}'> {1} </a> </li> ".format(os.path.join(WEBSERVER_ROOT, DUMP_DIR, 'debug', scene_id + '.html'), scene_id + '.html')
            fpaths += "</ul>"
            out += "<tr> <td> {0} </td> </tr>".format(fpaths)

        f_out = HTML_BASE.format(CSS_BASE, 'Index of Experiments', out)
        with open(dpath, 'w') as f:
            f.write(f_out)


    def dump_matching_as_html(self, rpath, dpath, method, key):
        
        obj = json.load(open(rpath, 'r'))

        out = "<table> <tr> <td> <strong> Matched Frame </strong> </td> <td> <strong> Rendered Frame </strong> </td> <td> <strong> Recording </strong> </td>  <td> <strong> Meta-data </strong>  </td> <td> <strong> Pixel Ratio </strong> </td> <td> <strong> Camera Direction Angle Difference (matched vs fixed viewpoint) </strong> </td> <td> <strong> Camera Center L2 Distance (matched vs fixed viewpoint)  </strong> </td> </tr>"

        obj = obj[key]

        for item in obj:
            ann_id = item['ann_id']
            scene_id = item['scene_id']
            if 'ann' in method:
                frame_idx = int(item['frame_id'])

            object_id = item['object_id']
            object_name = item['object_name']
            pixel_ratio = item['pixel_ratio']
            scene_video_link = 'http://aspis.cmpt.sfu.ca/datasets/scannet/public/v2/scans_extra/video/{}/{}.mp4'.format(scene_id, scene_id)
            if 'ann' in method:
                angle_to = round(item['angle_difference'], 2)
                l2_to = round(item['center_difference'], 2)
            out += "<tr>"
            if 'ann' in method:
                out += "<td width=360px> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;'/></td> ".format(os.path.join(WEBSERVER_ROOT, self.args.frames_symlink, scene_id, str(frame_idx) + '.jpg'))
            if 'ren' in method:
                out += "<td width=360px> <img src='{0}' loading='lazy' width='360' height='202.5' /></td> ".format(os.path.join(WEBSERVER_ROOT, self.args.renders_symlink, scene_id, scene_id + '-' + str(object_id) + '_' + str(ann_id) + '.png'))
            out += "<td width=360px> <img src='{0}' loading='lazy' width='360' height='202.5' /> </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.renders_symlink, scene_id, scene_id + '-' + str(object_id) + '_' + str(ann_id) + '.png'))
            out += "<td width=360px> <video controls='controls' preload='none' width='320' height='240' > <source src='{0}' type='video/mp4' your browser does not support video </video> </td> ".format(scene_video_link)
            if 'ann' in method:
                out += "<td> {0} </td>".format('{}|{}|{}|{}|'.format(scene_id, frame_idx, object_id, object_name))
            if 'ren' in method:
                out += "<td> {0} </td>".format('{}|{}|{}|{}|'.format(scene_id, object_id, ann_id, object_name))

            out += "<td> {0} </td>".format(pixel_ratio)
            if 'ann' in method:
                out += "<td> {0} </td>".format(angle_to)
                out += "<td> {0} </td>".format(l2_to)
            if 'ren' in method:
                out += "<td> {0} </td>".format('Not Assigned')
                out += "<td> {0} </td>".format('Not Assigned')
            out += "</tr>"

        out += "</table>"

        f_out = HTML_BASE.format(CSS_BASE, os.path.join(method, key), out)

        out_dir = os.path.join(self.args.dump_path, method)
        if not os.path.exists(out_dir):
            print("Making new directory: ", out_dir)
            os.makedirs(out_dir)

        with open(dpath, 'w') as f:
            f.write(f_out)

    def dump_debug_html(self):
        # scl_1 = list(set([item['scene_id'] for item in json.load(open(os.path.join(args.matches_path, 'debug_matches_train.json')))]))
        scl_2 = list(set([item['scene_id'] for item in json.load(open(os.path.join(args.matches_path, 'debug_matches_val.json')))]))
        scl = scl_2
        # m1 = json.load(open(os.path.join(args.matches_path, 'debug_matches_train.json')))
        m2 = json.load(open(os.path.join(args.matches_path, 'debug_matches_val.json')))
        m = m2
        scene_matches = {}
        # load
        for scene_id in scl:
            scene_matches[scene_id] = sorted([item for item in m if item['scene_id'] == scene_id], key= lambda x: (int(x['object_id']), int(x['ann_id'])))

        out_dir = os.path.join(self.args.dump_path, 'debug')
        if not os.path.exists(out_dir):
            print("Making new directory: ", out_dir)
            os.makedirs(out_dir)

        # dump
        num_debug_samples = 10
        for scene_id, items in scene_matches.items():
            out = "<table>"
            out += "<tr> <td> Rendered Frame </td>"
            out += "".join(["<td> {}'th best (direction angle) </td> <td> {}'th best (origin distance) </td>".format(i + 1, i + 1) for i in range(num_debug_samples)])
            out += "</tr>"

            print("Number of items:", len(items))
            for item in items:
                out += "<tr>"
                angles, angles_indices, distances, distances_indices = item['angles'], item['angles_indices'], item['distances'], item['distances_indices']
                object_id = item['object_id'] 
                ann_id = item['ann_id']
                object_name = item['object_name']
                out += "<td width=360px> <figure> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;'/> <figcaption> SceneID: {1} | ObjectID: {2} | ObjectName: {3} | AnnID: {4} </figcaption> </figure> </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.debug_rendered_frames_with_bbox, scene_id, scene_id + '-' + str(object_id) + '_' + str(ann_id) + '.png'), scene_id, object_id, object_name, ann_id)                
                for i in range(num_debug_samples):
                    frame_idx = int(angles_indices[i])
                    out += "<td width=360px> <figure> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;'/> <figcap> Frame Idx: {1} | Angle: {2} </figcap> </figure>  </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.frames_symlink, scene_id, str(frame_idx) + '.jpg'), str(frame_idx), str(round(angles[i], 2)))
                    frame_idx = int(distances_indices[i])
                    out += "<td width=360px> <figure> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;'/> <figcap> Frame Idx: {1} | L2: {2} </figcap> <figure> </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.frames_symlink, scene_id, str(frame_idx) + '.jpg'), str(frame_idx), str(round(distances[i], 2)))
                out += "</tr>"
            out += "</table> <br/> "
        
            f_out = HTML_BASE.format(CSS_BASE, "scene info", out)
            dpath = os.path.join(out_dir, scene_id + '.html')
            with open(dpath, 'w') as f:
                f.write(f_out)

    def dump_as_html(self, rpath, dpath, exp, key):
        """
            Dump a file from rpath (json) to 
            dpath (html) with proper formatting.
        """
        bbox_visualization = False
        if 'oracle' in exp:
            # print("Activated gt bbox visualization.")
            bbox_visualization = True
            drawn_bbox_dir = 'gt/{}/{}' # gt/det scene_id, frame_id

        if 'det' in exp:
            # print("Activated gt bbox visualization.")
            bbox_visualization = True
            drawn_bbox_dir = 'det/{}/{}' # gt/det scene_id, frame_id

        obj = json.load(open(rpath, 'r'))
        gt = json.load(open(rpath.replace('pred', 'corpus'), 'r'))

        if key == 'pred_train':
            matches = json.load(open(os.path.join(self.args.in_use_matches_path, 'matches_train.json')))
            scores_dict = json.load(open(os.path.join(self.args.output_path, exp, 'score_train.json')))
        elif key == 'pred_val':
            matches = json.load(open(os.path.join(self.args.in_use_matches_path, 'matches_val.json')))
            scores_dict = json.load(open(os.path.join(self.args.output_path, exp, 'score_val.json')))
        else:
            print("Unsupported keys.")
            exit(0)

        out = "<table> <tr> <td> <strong> Annotated Frame </strong> </td> <td> <strong> Rendered Frame </strong> </td> <td> <strong> Meta-data (SceneID, ObjectID, ObjectName) </strong>  </td> <td> <strong> Prediction </strong></td> <td> <strong> Ground-Truth </strong> </td> <td> <strong> BLEU_4 </strong></td> <td> <strong> CiDEr </strong> </td> <td> <strong> METEOR </strong> </td><td> <strong> ROUGE-L </strong> </td></tr>"
        idx = 0
        for ID, CAP in obj.items():
            out += "<tr>"
            fr = next(filter(lambda item: "|".join([item['scene_id'], item['object_id'], item['object_name']]) == ID, matches))
            if bbox_visualization:
                out += "<td width=360px> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;' /> </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.bbox_symlink, drawn_bbox_dir.format(ID.split('|')[0], str(fr['frame_idx']) + '.jpg')))
            else:
                out += "<td width=360px> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;' /> </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.frames_symlink, ID.split('|')[0], str(fr['frame_idx']) + '.jpg'))

                # out += "<td width=360px> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;' </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.frames_symlink, ID.split('|')[0], str(fr['frame_idx']) + '.jpg'))

            out += "<td width=360px> <figure> <img src='{0}' loading='lazy' style='width:360px;height:202.5px;'/> <figcaption> SceneID: {1} | ObjectID: {2} | ObjectName: {3} | AnnID: {4} </figcaption> </figure> </td> ".format(os.path.join(WEBSERVER_ROOT, self.args.renders_symlink, fr['scene_id'], fr['scene_id'] + '-' + str(fr['object_id']) + '_' + str(fr['ann_id']) + '.png'), fr['scene_id'], fr['object_id'], fr['object_name'], fr['ann_id'])                
            out += "<td> {0} </td>".format(ID)
            out += "<td> {0} </td>".format(CAP[0])
            out += "<td> <ul> {0} </ul> </td>".format("".join([' <li> {} </li>'.format(i) for i in gt[ID]]))

            scores = {
                "bleu_4": "<strong> {0} </strong>".format(round(float(scores_dict['bleu-4'][idx]) * 100, 2)),
                "cider": "<strong> {0} </strong>".format(round(float(scores_dict['cider'][idx]) * 100, 2)),
                "rouge": "<strong> {0} </strong>".format(round(float(scores_dict['rouge'][idx]) * 100, 2)),
                "meteor": "<strong> {0} </strong>".format(round(float(scores_dict['meteor'][idx]) * 100, 2))
            }
            out += "<td> <strong> {0} </strong> </td>".format(scores['bleu_4'])
            out += "<td> <strong> {0}  </strong> </td>".format(scores['cider'])
            out += "<td> <strong> {0}  </strong> </td>".format(scores['rouge'])
            out += "<td> <strong> {0}  </strong> </td>".format(scores['meteor'])
            # out += '<img src= "' + data[scene_id][j]["url"] + '" style="width:360px;height:202.5px;">'
            out += "</tr>"
            idx += 1

        out += "</table>"

        f_out = HTML_BASE.format(CSS_BASE, os.path.join(exp, key), out)

        out_dir = os.path.join(self.args.dump_path, exp)
        if not os.path.exists(out_dir):
            print("Making new directory: ", out_dir)
            os.makedirs(out_dir)

        with open(dpath, 'w') as f:
            f.write(f_out)


    def dump_htmls(self, dump_keys):
        """
            Loop through all experiments in 
            the output folder, get the train/val predictions
            and dump them to HTML files.
        """

        exp_names = [item for item in os.listdir(self.args.output_path) if not item.endswith('.json') and not "2020" in item]
        print("Target experiments: ", exp_names)
        matching_methods = [item for item in os.listdir(self.args.matching_stats) if not "avgs" in item] # directly json files
        matching_dump_keys = ['0', '5', '10', '15', '20', '25', '30', '50', '80', '95']
            
        if not os.path.isfile(self.args.index_file):
            print("Creating the table of experiments.")
            self.create_html_index(exp_names, dump_keys, [i.strip('.json') for i in matching_methods], matching_dump_keys)

        if self.args.debug:
            print("dumping debug html files.")
            self.dump_debug_html()

        print("Dumping matching methods...")

        for method in matching_methods:
            for k in matching_dump_keys:
                rpath = os.path.join(self.args.matching_stats, method)
                dpath = os.path.join(self.args.dump_path, method.strip('.json'), k + '.html')

                if not os.path.isfile(dpath):
                    self.dump_matching_as_html(rpath, dpath, method.strip('.json'), k)
                else:
                    continue;

        print("Dumping experiments...")
        for exp in exp_names:
            for k in dump_keys:
                rpath = os.path.join(self.args.output_path, exp, k + '.json')
                dpath = os.path.join(self.args.dump_path, exp, k + '.html')

                if not os.path.isfile(dpath):
                    self.dump_as_html(rpath, dpath, exp, k)
                else:
                    continue;


                
def parse_arguments():

    ap = argparse.ArgumentParser()
    ap.add_argument('--output_path', nargs='?', default='/local-scratch/code/scan2cap_codebase/Scan2Cap/outputs')
    ap.add_argument('--dump_path', nargs='?', default='/project/3dlg-hcvc/scan2cap/www/html_dumps')
    ap.add_argument('--index_file', nargs='?', default='/project/3dlg-hcvc/scan2cap/www/index.html')
    ap.add_argument('--matches_path', nargs='?', default='/local-scratch/code/scan2cap_extracted/match-based/vp_matching')
    ap.add_argument('--in_use_matches_path', nargs='?', default='/project/3dlg-hcvc/gholami/vp_matching')
    ap.add_argument('--frames_symlink', nargs='?', default='sens2frame_output')
    ap.add_argument('--bbox_symlink', nargs='?', default='bbox_visualized')
    ap.add_argument('--renders_symlink', nargs='?', default='visualized_boxes_renders/gt')
    ap.add_argument('--matching_stats', nargs='?', default='/local-scratch/code/scan2cap_extracted/match-based/matching_stats/')
    ap.add_argument('--plots_symlink', nargs='?', default='plots')
    ap.add_argument('--debug_matched_frames_with_bbox', nargs='?', default='visualized_boxes/matched/gt')
    ap.add_argument('--debug_rendered_frames_with_bbox', nargs='?', default='visualized_boxes/rendered/gt')
    ap.add_argument('--debug', action='store_true', default=False)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    html_visualizer = Scan2CapHTMLVisualizer(args)
    try:
        os.remove(args.index_file)
        shutil.rmtree(args.dump_path)
    except FileNotFoundError:
        pass

    print("Removed existing dumps.")
    dump_keys = ['pred_val', 'pred_train']
    html_visualizer.dump_htmls(dump_keys)
