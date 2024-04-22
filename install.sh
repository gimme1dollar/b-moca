#/bin/bash

python asset/environments/set_up.py --avd_name 0 --mode "train" --port 5548
python asset/environments/set_up.py --avd_name 0 --mode "test" --port 5560

echo "Done"