from es import ES
import lib

def main():
    es_sphere = ES(5, 1, lib.sphere, lb=[-5, -5], ub=[5, 5])
    best = es_sphere.optimize()
    print(f'MIN: {best}')

if __name__ == '__main__':
    main()
