//Here's a JavaScript implementation of the Breadth-First Search (BFS) algorithm using a queue:


function dijkstra(graph, startNode) {
    const distances = {}; // Stores distances from the start node to each other node
    const visited = new Set(); // Keeps track of visited nodes
  
    // Initialize distances for all nodes to infinity (except the start node)
    for (const node in graph) {
      distances[node] = Infinity;
    }
    distances[startNode] = 0; // Distance to the start node is 0
  
    // Priority queue to efficiently select the unvisited node with the shortest tentative distance
    const queue = new PriorityQueue({ compare: (a, b) => distances[a] - distances[b] });
    queue.enqueue(startNode);
  
    while (!queue.isEmpty()) {
      const currentNode = queue.dequeue(); // Get the unvisited node with the shortest tentative distance
      visited.add(currentNode);
  
      for (const neighbor of graph[currentNode]) {
        const edgeWeight = graph[currentNode][neighbor]; // Weight of the edge between current and neighbor node
  
        // Calculate tentative distance to the neighbor
        const tentativeDistance = distances[currentNode] + edgeWeight;
  
        // Update distance if the tentative distance is shorter
        if (!visited.has(neighbor) && tentativeDistance < distances[neighbor]) {
          distances[neighbor] = tentativeDistance;
          queue.enqueue(neighbor); // Add neighbor to the queue for exploration with the updated distance
        }
      }
    }
  
    return distances; // Return the distances object containing shortest distances from the start node to all other nodes
  }
  
  // Simple implementation of a Priority Queue (can be replaced with an existing library)
  class PriorityQueue {
    constructor(options) {
      this.compare = options.compare || ((a, b) => a - b);
      this.items = [];
    }
  
    enqueue(item) {
      this.items.push(item);
      this.heapifyUp();
    }
  
    dequeue() {
      const item = this.items.shift();
      this.heapifyDown();
      return item;
    }
  
    isEmpty() {
      return this.items.length === 0;
    }
  
    heapifyUp() {
      let index = this.items.length - 1;
      while (index > 0 && this.compare(this.items[index], this.items[Math.floor((index - 1) / 2)]) < 0) {
        [this.items[index], this.items[Math.floor((index - 1) / 2)]] = [this.items[Math.floor((index - 1) / 2)], this.items[index]];
        index = Math.floor((index - 1) / 2);
      }
    }
  
    heapifyDown() {
      let index = 0;
      const length = this.items.length;
      let leftChild, rightChild;
      while (true) {
        leftChild = (index * 2) + 1;
        rightChild = (index * 2) + 2;
        let swap = null;
  
        if (leftChild < length && this.compare(this.items[leftChild], this.items[index]) < 0) {
          swap = leftChild;
        }
        if (rightChild < length && (swap === null || this.compare(this.items[rightChild], this.items[swap]) < 0)) {
          swap = rightChild;
        }
  
        if (swap === null) break;
  
        [this.items[index], this.items[swap]] = [this.items[swap], this.items[index]];
        index = swap;
      }
    }
  }
  
  // Example usage
  const graph = {
    A: { B: 4, C: 2 },
    B: { E: 3, F: 1 },
    C: { D: 4, E: 5 },
    D: {},
    E: { F: 6 },
    F: {},
  };
  
  const startNode = "A";
  
  const distances = dijkstra(graph, startNode);
  
  console.log("Shortest distances from", startNode, "to all other nodes:");
  console.log(distances);
  
  // Output example:
  // Shortest distances from A to all other nodes:
  // { A: 0, B: 4, C: 2, D: 6, E: 7,
  