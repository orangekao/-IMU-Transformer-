EXECUTION_COUNT=30
cd ./Data_preprocessing

bash all_preprocess.sh

cd ..

Learning_rate=(3e-4 5e-4 7e-4 5e-3)

for idx in "${Learning_rate[@]}"; do

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "------------------Train【$i】------------------"
        python train.py --lr $idx --overlap 90 --slide 108 --category 8 --epochs 200 --times $i --batch 128 --single_in false 
    done

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "-------------------Test【$i】------------------"
        python test.py  --lr $idx --overlap 90 --slide 108 --category 8 --times $i --batch 128 --single_in false
    done

    python acc_analysis.py --lr $idx --overlap 80 --slide 108

done

# 如果上面的slide改 這裡面的也要改
python plot_box.py
