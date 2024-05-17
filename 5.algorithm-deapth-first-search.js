//Here's a JavaScript implementation of the Depth-First Search (DFS) algorithm using recursion:


function dfs(graph, startNode) {
    const visited = new Set(); // Keeps track of visited nodes
  
    function dfsHelper(currentNode) {
      visited.add(currentNode); // Mark the current node as visited
      console.log("Visiting:", currentNode); // You can replace this with your processing logic
  
      const neighbors = graph[currentNode]; // Get the neighbors of the current node
  
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) { // Check if neighbor is not yet visited
          dfsHelper(neighbor); // Recursive call to explore the neighbor
        }
      }
    }
  
    dfsHelper(startNode); // Initial call to start the recursion
  }
  
  // Example usage:
  
  const graph = {
    A: ["B", "C"],
    B: ["D", "E"],
    C: ["F"],
    D: [],
    E: ["F"],
    F: [],
  };
  
  const startingNode = "A";
  
  dfs(graph, startingNode);
  