# pretreat train set
python pretreatment_grew.py --input_path "/GREW" --output_path "/GREW-64-44-pkl" --img_h 64 --img_w 44 --subset "train"
# pretreat test-gallery set
python pretreatment_grew.py --input_path "/GREW" --output_path "/GREW-64-44-pkl" --img_h 64 --img_w 44 --subset "test/gallery"
# pretreat test-probe set
python pretreatment_grew_probe.py --input_path "/GREW" --output_path "/GREW-64-44-pkl" --img_h 64 --img_w 44