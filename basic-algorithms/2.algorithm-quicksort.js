function quickSort(arr) {
    if (arr.length <= 1) {
      return arr; // Base case: already sorted (or empty)
    }
  
    // Choose a pivot element (here, the last element)
    let pivot = arr[arr.length - 1];
    let left = [];
    let right = [];
  
    // Partition the array
    for (let i = 0; i < arr.length - 1; i++) {
      if (arr[i] < pivot) {
        left.push(arr[i]);
      } else {
        right.push(arr[i]);
      }
    }
  
    // Recursively sort the left and right sub-arrays
    return [...quickSort(left), pivot, ...quickSort(right)];
  }
  
  // Example usage:
  const numbers = [5, 3, 8, 2, 1, 4];
  const sortedNumbers = quickSort(numbers);
  
  console.log(sortedNumbers); // Output: [1, 2, 3, 4, 5, 8]
  