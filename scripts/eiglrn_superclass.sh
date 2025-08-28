echo "Gaussian Kernel @ CIFAR10 Vehicle vs Animal"
python -u scripts/hehe_eigenlearning.py 3 vehicle
echo "Gaussian Kernel @ CIFAR10 Domesticated vs Wild"
python -u scripts/hehe_eigenlearning.py 3 domesticated
echo "Gaussian Kernel @ SVHN Odd vs Even"
python -u scripts/hehe_eigenlearning.py 5 evenodd
echo "Gaussian Kernel @ SVHN Looped numerals vs Others"
python -u scripts/hehe_eigenlearning.py 5 loops

echo "Laplace Kernel @ CIFAR10 Vehicle vs Animal"
python -u scripts/hehe_eigenlearning.py 4 vehicle
echo "Laplace Kernel @ CIFAR10 Domesticated vs Wild"
python -u scripts/hehe_eigenlearning.py 4 domesticated
echo "Laplace Kernel @ SVHN Odd vs Even"
python -u scripts/hehe_eigenlearning.py 6 evenodd
echo "Laplace Kernel @ SVHN Looped numerals vs Others"
python -u scripts/hehe_eigenlearning.py 6 loops

echo "All experiments completed."