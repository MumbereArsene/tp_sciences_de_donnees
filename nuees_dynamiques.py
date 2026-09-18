import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# JEU DE DONNEES
# ============================================================

X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],

    [8, 8],
    [9, 7],
    [8, 6],
    [7, 7],

    [4, 5],
    [5, 4],
    [6, 5],
    [5, 6]
])

# Nombre de groupes
K = 3


# ============================================================
# CALCUL DE LA DISTANCE EUCLIDIENNE
# ============================================================

def distance_euclidienne(point, centre):
    """
    Calcule la distance euclidienne entre un point
    et un centre.
    """

    difference = point - centre

    distance = np.sqrt(
        np.sum(difference ** 2)
    )

    return distance


# ============================================================
# INITIALISATION DES CENTRES
# ============================================================

def initialiser_centres(X, k):
    """
    Selectionne aleatoirement k observations
    comme centres initiaux.
    """

    indices = np.random.choice(
        len(X),
        k,
        replace=False
    )

    centres = X[indices].astype(float)

    return centres


# ============================================================
# AFFECTATION DES OBSERVATIONS
# ============================================================

def affecter_points(X, centres):
    """
    Affecte chaque observation au centre
    le plus proche.
    """

    groupes = []

    for point in X:

        distances = []

        for centre in centres:

            distance = distance_euclidienne(
                point,
                centre
            )

            distances.append(distance)

        groupe = np.argmin(distances)

        groupes.append(groupe)

    return np.array(groupes)


# ============================================================
# RECALCUL DES CENTRES
# ============================================================

def recalculer_centres(X, groupes, k):
    """
    Recalcule les centres des groupes
    en utilisant la moyenne des observations.
    """

    nouveaux_centres = []

    for i in range(k):

        points = X[groupes == i]

        if len(points) > 0:

            centre = np.mean(
                points,
                axis=0
            )

        else:

            centre = np.zeros(
                X.shape[1]
            )

        nouveaux_centres.append(
            centre
        )

    return np.array(
        nouveaux_centres
    )


# ============================================================
# ALGORITHME DES NUEES DYNAMIQUES
# ============================================================

def nuees_dynamiques(
    X,
    k,
    max_iterations=100
):
    """
    Implementation From Scratch de l'algorithme
    des nuees dynamiques.

    Parametres
    ----------
    X : tableau numpy
        Jeu de donnees.

    k : int
        Nombre de groupes.

    max_iterations : int
        Nombre maximal d'iterations.

    Retour
    ------
    groupes : numpy.ndarray
        Groupe de chaque observation.

    centres : numpy.ndarray
        Centres finaux.

    iterations : int
        Nombre d'iterations executees.
    """

    centres = initialiser_centres(
        X,
        k
    )

    for iteration in range(
        max_iterations
    ):

        anciens_centres = centres.copy()

        # Etape 1 : affectation
        groupes = affecter_points(
            X,
            centres
        )

        # Etape 2 : recalcul des centres
        centres = recalculer_centres(
            X,
            groupes,
            k
        )

        # Etape 3 : verification de la convergence
        if np.allclose(
            anciens_centres,
            centres
        ):
            break

    return (
        groupes,
        centres,
        iteration + 1
    )


# ============================================================
# CALCUL DE L'INERTIE
# ============================================================

def inertie(
    X,
    groupes,
    centres
):
    """
    Calcule l'inertie intra-classe.
    """

    total = 0.0

    for i in range(len(X)):

        centre = centres[
            groupes[i]
        ]

        distance = distance_euclidienne(
            X[i],
            centre
        )

        total += distance ** 2

    return total


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    # Graine aleatoire pour obtenir
    # des resultats reproductibles
    np.random.seed(42)

    # Execution de l'algorithme
    groupes, centres, iterations = (
        nuees_dynamiques(
            X,
            K
        )
    )

    # Calcul de l'inertie
    score = inertie(
        X,
        groupes,
        centres
    )

    # Affichage des resultats
    print("=" * 50)
    print("ALGORITHME DES NUEES DYNAMIQUES")
    print("=" * 50)

    print("\nJeu de donnees :")
    print(X)

    print("\nNombre de groupes :", K)

    print("\nGroupes obtenus :")
    print(groupes)

    print("\nCentres finaux :")
    print(centres)

    print("\nNombre d'iterations :", iterations)

    print("\nInertie :", score)

    print("\nAffectation des observations :")

    for i in range(len(X)):

        print(
            f"Observation {i + 1}: "
            f"{X[i]} -> Groupe "
            f"{groupes[i] + 1}"
        )

    # ========================================================
    # VISUALISATION
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    for groupe in range(K):

        points = X[
            groupes == groupe
        ]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            label=f"Groupe {groupe + 1}"
        )

    # Affichage des centres
    plt.scatter(
        centres[:, 0],
        centres[:, 1],
        marker="X",
        s=200,
        label="Centres"
    )

    plt.xlabel("Variable X1")
    plt.ylabel("Variable X2")

    plt.title(
        "Resultat des nuees dynamiques"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


# ============================================================
# LANCEMENT DU PROGRAMME
# ============================================================

if __name__ == "__main__":
    main()