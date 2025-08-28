echo "Gaussian Kernel @ Gaussian data"
python -u scripts/hehe_eigenlearning.py 1 powerlaws
echo "Laplace Kernel @ Gaussian data"
python -u scripts/hehe_eigenlearning.py 2 powerlaws
echo "Gaussian Kernel @ CIFAR10"
python -u scripts/hehe_eigenlearning.py 3 powerlaws
echo "Laplace Kernel @ CIFAR10"
python -u scripts/hehe_eigenlearning.py 4 powerlaws
echo "All experiments completed."