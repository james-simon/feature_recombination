echo "Gaussian Kernel @ Gaussian data"
python -u scripts/hehe_eigenlearning.py 1 monomials
echo "Laplace Kernel @ Gaussian data"
python -u scripts/hehe_eigenlearning.py 2 monomials
echo "Gaussian Kernel @ CIFAR10"
python -u scripts/hehe_eigenlearning.py 3 monomials
echo "Laplace Kernel @ CIFAR10"
python -u scripts/hehe_eigenlearning.py 4 monomials
echo "All experiments completed."