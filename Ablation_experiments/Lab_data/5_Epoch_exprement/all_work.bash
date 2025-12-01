EXECUTION_COUNT=30
cd ./Data_preprocessing

bash all_preprocess.sh

cd ..

epochs=(50 100 150 200)
# epochs=(150)

for idx in "${epochs[@]}"; do

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "------------------Train【$i】------------------"
        python train.py --lr 5e-4 --overlap 90 --slide 108 --category 8 --epochs $idx --times $i --batch 16 --single_in false 
    done

    for ((i=1; i<=$EXECUTION_COUNT; i++)); do
        echo "-------------------Test【$i】------------------"
        python test.py  --epochs $idx --lr 5e-4 --overlap 90 --slide 108 --category 8 --times $i --batch 16 --single_in false
    done

    python acc_analysis.py --epochs $idx --overlap 90 --slide 108

done

# 如果上面的slide改 這裡面的也要改
python plot_box.py
