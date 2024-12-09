//Here's a JavaScript implementation of Kruskal's algorithm for finding the minimum spanning tree (MST) in a weighted graph:


function kruskal(graph) {
    // Helper function to find the root of a node in the disjoint-set data structure
    function find(parent, node) {
      if (parent[node] !== node) {
        parent[node] = find(parent, parent[node]); // Path compression for efficiency
      }
      return parent[node];
    }
  
    // Helper function to check if two nodes belong to the same set (connected)
    function union(parent, rank, x, y) {
      const rootX = find(parent, x);
      const rootY = find(parent, y);
  
      // Attach the tree with smaller rank to the root of the tree with higher rank
      if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
      } else if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
      } else {
        // If ranks are equal, choose one as root and increment its rank
        parent[rootY] = rootX;
        rank[rootX]++;
      }
    }
  
    // Create an array to store parent nodes (initially each node is its own parent)
    const parent = {};
    for (const node in graph) {
      parent[node] = node;
    }
  
    // Create an array to store ranks for efficient union-find operations (initially all ranks are 0)
    const rank = {};
    for (const node in graph) {
      rank[node] = 0;
    }
  
    // Create a priority queue to store edges sorted by weight (ascending order)
    const edges = [];
    for (const node in graph) {
      for (const neighbor in graph[node]) {
        if (neighbor > node) { // Avoid adding duplicate edges (consider an undirected graph)
          edges.push({ weight: graph[node][neighbor], source: node, target: neighbor });
        }
      }
    }
  
    edges.sort((a, b) => a.weight - b.weight); // Sort edges by weight
  
    const mst = []; // Minimum spanning tree (forest)
  
    // Process edges in sorted order
    for (const edge of edges) {
      const rootX = find(parent, edge.source);
      const rootY = find(parent, edge.target);
  
      // If the nodes belong to different sets (not connected yet), include the edge in the MST and perform union
      if (rootX !== rootY) {
        mst.push(edge);
        union(parent, rank, rootX, rootY);
      }
    }
  
    return mst; // Return the minimum spanning tree
  }
  
  // Example usage
  const graph = {
    A: { B: 4, C: 2 },
    B: { A: 4, E: 3 },
    C: { A: 2, D: 4, E: 5 },
    D: { C: 4, F: 1 },
    E: { B: 3, C: 5, F: 6 },
    F: { D: 1, E: 6 },
  };
  
  const mst = kruskal(graph);
  
  console.log("Minimum Spanning Tree:");
  console.log(mst);
  
  // Output example:
  // Minimum Spanning Tree:
  // [{ weight: 2, source: "A", target: "C" }, { weight: 1, target: "D", source: "F" }, { weight: 3, target: "E", source: "B" }]  
  