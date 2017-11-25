from es import ES
import lib

def main():
    es_sphere = ES(100, 100, lib.sphere, lb=[-5, -5], ub=[5, 5])
    es_ackley = ES(100, 100, lib.ackley, lb=[-20, -20], ub=[20, 20])
    es_rastrigin = ES(100, 100, lib.rastrigin, lb=[-5, -5], ub=[5, 5])

    best = es_sphere.optimize()
    print('Sphere')
    print(f'MIN: {best}')

    best = es_ackley.optimize()
    print('Ackley')
    print(f'MIN: {best}')

    best = es_rastrigin.optimize()
    print('Rastrigin')
    print(f'MIN: {best}')

if __name__ == '__main__':
    main()
