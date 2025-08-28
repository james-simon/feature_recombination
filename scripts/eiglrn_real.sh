echo "Gaussian Kernel @ CIFAR10"
python -u scripts/hehe_eigenlearning.py 3 original
echo "Laplace Kernel @ CIFAR10"
python -u scripts/hehe_eigenlearning.py 4 original
echo "Gaussian Kernel @ SVHN"
python -u scripts/hehe_eigenlearning.py 5 original
echo "Laplace Kernel @ SVHN"
python -u scripts/hehe_eigenlearning.py 6 original
echo "All experiments completed."