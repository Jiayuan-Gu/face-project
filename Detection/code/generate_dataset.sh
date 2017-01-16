for (( i = 0; i < 61; i++ )); do
	python generate_dataset.py $i &#statements
done
