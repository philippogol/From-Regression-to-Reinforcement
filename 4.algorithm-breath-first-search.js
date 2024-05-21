//Here's a JavaScript implementation of the Breadth-First Search (BFS) algorithm using a queue:

function bfs(graph, startNode) {
  const visited = new Set(); // Set to store visited nodes
  const queue = []; // Queue to store nodes to be explored

  queue.push(startNode);
  visited.add(startNode);

  while (queue.length > 0) {
    const currentNode = queue.shift(); // Get the first element from the queue
    console.log("Visiting:", currentNode); // Visit the current node (you can modify this to perform actions)

    const neighbors = graph[currentNode]; // Get the neighbors of the current node
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
}

// Example usage (assuming a graph is defined as an adjacency list)
const graph = {
  A: ["B", "C"],
  B: ["D", "E"],
  C: ["F"],
  D: [],
  E: ["F"],
  F: [],
};

const startingNode = "A";
bfs(graph, startingNode);
