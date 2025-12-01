overlap=(80)
# overlap=(0 50 78 90)

for idx in "${overlap[@]}";
do

    echo "--------preprocessing--------"
    python preprocessing.py --axis 3 --slide 108 --overlap $idx

    echo "------------SMOTE------------"
    python augment.py --axis 3 --slide 108 --overlap $idx

    echo "-------------FFT-------------"
    python FFT.py --slide 108 --overlap $idx
done