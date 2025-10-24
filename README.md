### EX7 Implementation of Link Analysis using HITS Algorithm
### DATE: 24.10.2025
### NAME: EZHIL NEVEDHA K
### REG.NO: 212223230055
### AIM: To implement Link Analysis using HITS Algorithm in Python.
### Description:
<div align = "justify">
The HITS (Hyperlink-Induced Topic Search) algorithm is a link analysis algorithm used to rank web pages. It identifies authority and hub pages 
in a network of web pages based on the structure of the links between them.

### Procedure:
1. ***Initialization:***
    <p>    a) Start with an initial set of authority and hub scores for each page.
    <p>    b) Typically, initial scores are set to 1 or some random values.
  
2. ***Construction of the Adjacency Matrix:***
    <p>    a) The web graph is represented as an adjacency matrix where each row and column correspond to a web page, and the matrix elements denote the presence or absence of links between pages.
    <p>    b) If page A has a link to page B, the corresponding element in the adjacency matrix is set to 1; otherwise, it's set to 0.

3. ***Iterative Updates:***
    <p>    a) Update the authority scores based on the hub scores of pages pointing to them and update the hub scores based on the authority scores of pages they point to.
    <p>    b) Calculate authority scores as the sum of hub scores of pages pointing to the given page.
    <p>    c) Calculate hub scores as the sum of authority scores of pages that the given page points to.

4. ***Normalization:***
    <p>    a) Normalize authority and hub scores to prevent them from becoming too large or small.
    <p>    b) Normalize by dividing by their Euclidean norms (L2-norm).

5. ***Convergence Check:***
    <p>    a) Check for convergence by measuring the change in authority and hub scores between iterations.
    <p>    b) If the change falls below a predefined threshold or the maximum number of iterations is reached, the algorithm stops.

6. ***Visualization:***
    <p>    Visualize using bar chart to represent authority and hub scores.

### Program:

```python
import numpy as np
import matplotlib.pyplot as plt

def hits_algorithm(adjacency_matrix, max_iterations=100, tol=1.0e-6):
    num_nodes = len(adjacency_matrix)
    authority_scores = np.ones(num_nodes)
    hub_scores = np.ones(num_nodes)
    
    for i in range(max_iterations):
        old_authority_scores = authority_scores.copy()
        old_hub_scores = hub_scores.copy()
        
        authority_scores = np.dot(adjacency_matrix.T, old_hub_scores)
        auth_norm = np.linalg.norm(authority_scores)
        if auth_norm > 0:
            authority_scores /= auth_norm
        
        hub_scores = np.dot(adjacency_matrix, old_authority_scores)
        hub_norm = np.linalg.norm(hub_scores)
        if hub_norm > 0:
            hub_scores /= hub_norm
        
        authority_diff = np.sum(np.abs(authority_scores - old_authority_scores))
        hub_diff = np.sum(np.abs(hub_scores - old_hub_scores))
        
        if authority_diff < tol and hub_diff < tol:
            print(f"\nIterations stopped at {i+1}")
            break
            
    return authority_scores, hub_scores

adj_matrix = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0]
])

authority, hub = hits_algorithm(adj_matrix)

print("\nFinal Scores:")
for i in range(len(authority)):
    print(f"Node {i+1}: Authority Score = {authority[i]:.4f}, Hub Score = {hub[i]:.4f}")

sorted_authority = sorted(enumerate(authority, start=1), key=lambda x: x[1], reverse=True)
sorted_hub = sorted(enumerate(hub, start=1), key=lambda x: x[1], reverse=True)

print("\nRanking based on Authority Scores:")
for rank, (node, score) in enumerate(sorted_authority, start=1):
    print(f"Rank {rank}: Node {node} → {score}")

print("\nRanking based on Hub Scores:")
for rank, (node, score) in enumerate(sorted_hub, start=1):
    print(f"Rank {rank}: Node {node} → {score}")

nodes = np.arange(len(authority))
bar_width = 0.35
plt.figure(figsize=(10, 7))
plt.bar(nodes - bar_width/2, authority, bar_width, label='Authority', color='royalblue')
plt.bar(nodes + bar_width/2, hub, bar_width, label='Hub', color='seagreen')
plt.xlabel('Nodes')
plt.ylabel('Normalized Scores')
plt.title('HITS Algorithm: Authority and Hub Scores')
plt.xticks(nodes, [f'Node {i+1}' for i in nodes])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

### Output:
<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/68e091af-80f5-48e0-8a6d-4dfadca5efe0" />
<img width="1016" height="703" alt="image" src="https://github.com/user-attachments/assets/1da3f696-ce6b-4d69-88dd-d729f531e268" />


### Result:
Hence,Successfully implemented Link Analysis using HITS Algorithm in Python.
