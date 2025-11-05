#!/usr/bin/env python3
"""
Tarea 2: Propagación en redes con GUILD y DIAMOnD
Patricia Rodríguez Lidueña

Este script implementa dos métodos de propagación en redes biológicas:
- GUILD (NetProp): difunde información desde los genes semilla por la red.
- DIAMOnD: añade genes al módulo según sus conexiones con las semillas.

El programa se ejecuta por línea de comandos y guarda los resultados en formato CSV.
"""

import argparse
import pandas as pd
import networkx as nx
import numpy as np
import re


def load_network(network_file):
    """
    Carga la red de interacción desde un archivo.
    Acepta tanto redes con 2 columnas (sin peso) como con 3 (con peso).
    """
    edges = []
    with open(network_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detectar si hay 2 o 3 columnas separadas por coma o espacio
            parts = re.split(r'[\s,]+', line)
            if len(parts) == 2:
                source, target = parts
                weight = 1.0  # peso por defecto
            elif len(parts) >= 3:
                source, weight, target = parts[0], float(parts[1]), parts[2]
            else:
                continue
            edges.append((source, target, float(weight)))

    # Crear grafo no dirigido con networkx
    G = nx.Graph()
    for source, target, weight in edges:
        G.add_edge(str(source), str(target), weight=weight)

    print(f"Red cargada: {len(G.nodes())} nodos y {len(G.edges())} aristas")
    return G


def load_seeds(seed_file, G):
    """
    Carga los genes semilla desde un archivo de texto y
    comprueba cuáles están presentes en la red.
    """
    seeds = [line.strip() for line in open(seed_file) if line.strip()]
    seeds_in_network = [s for s in seeds if s in G.nodes()]
    missing = [s for s in seeds if s not in G.nodes()]
    if missing:
        print(f"Advertencia: algunas semillas no están en la red: {missing}")
    print(f"Semillas utilizadas: {seeds_in_network}")
    return seeds_in_network


def guild_propagation(G, seeds, alpha=0.7, max_iter=100, tol=1e-6):
    """
    Implementa el algoritmo GUILD (NetProp).
    Usa un random walk con reinicio desde los genes semilla.
    alpha controla la probabilidad de reiniciar hacia las semillas.
    """
    # Inicializar vector de probabilidad
    p0 = {n: 0 for n in G.nodes()}
    for s in seeds:
        if s in p0:
            p0[s] = 1 / len(seeds)

    # Crear matriz de adyacencia normalizada
    A = nx.to_numpy_array(G)
    D = A.sum(axis=1)
    D_inv = 1 / D
    D_inv[D_inv == float("inf")] = 0
    W = (A.T * D_inv).T

    nodes = list(G.nodes())
    p_vec = np.array([p0[n] for n in nodes])

    # Propagación iterativa
    for i in range(max_iter):
        new_p = (1 - alpha) * np.dot(W.T, p_vec) + alpha * np.array([p0[n] for n in nodes])
        diff = np.linalg.norm(new_p - p_vec, 1)
        p_vec = new_p
        if diff < tol:
            print(f"Convergencia alcanzada en {i+1} iteraciones (diff={diff:.2e})")
            break

    results = pd.DataFrame({'node': nodes, 'score': p_vec})
    results.sort_values('score', ascending=False, inplace=True)
    return results


def diamond_algorithm(G, seeds, n_iterations=50):
    """
    Implementa el algoritmo DIAMOnD.
    Añade genes al módulo de enfermedad según el número de conexiones
    con las semillas iniciales.
    """
    disease_genes = set(seeds)
    candidate_scores = {}

    for i in range(n_iterations):
        scores = {}
        for node in G.nodes():
            if node not in disease_genes:
                neighbors = set(G.neighbors(node))
                k = len(neighbors & disease_genes)  # conexiones con el módulo actual
                scores[node] = k

        if not scores:
            break

        # Añadir el nodo más conectado al módulo
        new_gene = max(scores, key=scores.get)
        candidate_scores[new_gene] = scores[new_gene]
        disease_genes.add(new_gene)

    results = pd.DataFrame(list(candidate_scores.items()), columns=['node', 'score'])
    results.sort_values('score', ascending=False, inplace=True)
    return results


def main():
    """
    Ejecuta el script desde la línea de comandos.
    Permite elegir el algoritmo (GUILD o DIAMOnD) y los archivos de entrada y salida.
    """
    parser = argparse.ArgumentParser(description="Network Propagation (GUILD / DIAMOnD)")
    parser.add_argument('--network', required=True, help='Archivo de red')
    parser.add_argument('--seeds', required=True, help='Archivo de semillas')
    parser.add_argument('--output', required=True, help='Archivo CSV de salida')
    parser.add_argument('--algorithm', choices=['guild', 'diamond'], default='guild',
                        help='Algoritmo a ejecutar (guild o diamond)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Número de iteraciones (solo para DIAMOnD)')
    args = parser.parse_args()

    print("Cargando red...")
    G = load_network(args.network)
    print("Cargando semillas...")
    seeds = load_seeds(args.seeds, G)

    if args.algorithm == 'guild':
        print("Ejecutando propagación con GUILD (NetProp)...")
        results = guild_propagation(G, seeds)
    else:
        print(f"Ejecutando algoritmo DIAMOnD ({args.iterations} iteraciones)...")
        results = diamond_algorithm(G, seeds, n_iterations=args.iterations)

    # Guardar resultados
    results[['node', 'score']].to_csv(args.output, index=False, sep='\t', header=True)
    print(f"Resultados guardados en {args.output}")
    print("Top 10 resultados:")
    print(results.head(10))


if __name__ == "__main__":
    main()
