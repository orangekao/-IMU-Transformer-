EXECUTION_COUNT=30
cd ./Data_preprocessing

bash all_preprocess.sh

cd ..
overlap=(80)
# overlap=(0 50 78 90)

for idx in "${overlap[@]}"; do

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "------------------Train【$i】------------------"
        python train.py --overlap $idx --slide 108 --category 8 --epochs 200 --times $i --batch 128 --single_in false 
    done

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "-------------------Test【$i】------------------"
        python test.py  --overlap $idx --slide 108 --category 8 --times $i --batch 128 --single_in false
    done

    python acc_analysis.py --overlap $idx --slide 108

done

# 如果上面的slide改 這裡面的也要改
python plot_box.py
