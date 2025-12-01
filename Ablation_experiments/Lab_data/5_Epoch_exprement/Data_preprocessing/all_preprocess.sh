echo "--------preprocessing--------"
python preprocessing.py --axis 3 --slide 108 --overlap 90

echo "------------SMOTE------------"
python augment.py --axis 3 --slide 108 --overlap 90

echo "-------------FFT-------------"
python FFT.py
