
slid=(64 72 84 96 108)

for idx in "${slid[@]}";
do

    echo "--------preprocessing--------"
    python preprocessing.py --axis 3 --slide $idx --overlap 50

    echo "------------SMOTE------------"
    python augment.py --axis 3 --slide $idx

    echo "-------------FFT-------------"
    python FFT.py --slide $idx --overlap 50
done