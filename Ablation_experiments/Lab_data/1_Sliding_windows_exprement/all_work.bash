EXECUTION_COUNT=30
cd ./Data_preprocessing

bash all_preprocess.sh

cd ..

slide=(64 72 84 96 108)

for idex in "${slide[@]}"; do

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "------------------Train【$i】------------------"
        python train.py --slide $idex --category 8 --epochs 200 --times $i --batch 64 --single_in false 
    done

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "-------------------Test【$i】------------------"
        python test.py  --slide $idex --category 8 --times $i --batch 64 --single_in false
    done

    python acc_analysis.py --slide $idex 

done

# 如果上面的slide改 這裡面的也要改
python plot_box.py
