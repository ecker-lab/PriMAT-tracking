#! /bin/sh

cd src

python demo.py mot --load_tracking_model "../models/derek/model_last.pth"\
            --conf_thres 0.02\
            --det_thres 0.6\
            --new_overlap_thres 0.85\
            --sim_thres 0.8\
            --input_video ../videos/derek/Imag0002.mp4\
            --output_root ../videos/derek\
            --output_name "Imag0002_output"\
            --store_opt\
            --line_thickness 2\
            --arch hrnet_32\
            --output_format video\
            --reid_cls_names macaque\
            --proportion_iou 0.2\
            --use_gc\
            --gc_cls_names FA330,FA331,FB350,FB353,FB355,FC360,FD370,FG400,FH412,FK431,FL11,FM189,FM21,FO30,FO31,FP227,FR52,FT268,FT274,FT277,FT60,FT61,FZ104,FZ320,MA115,MA118,MA337,MA340,MC361,MD370,ME380,MF391,MG403,MH410,MH412,MH416,MI421,MI422,ML10,ML12,MM189,MM21,MO202,MP40,MR247,MR52,MR55,MR56,MT60,MU282,MV72,MV73,MV74,MW283,MX87,MX95,MZ102,MZ321\




cd ..
