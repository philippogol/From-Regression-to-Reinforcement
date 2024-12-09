function mergeSort(arr) {
    if (arr.length <= 1) {
      return arr; // Base case: already sorted (or empty)
    }
  
    // Divide the array into two halves
    const middle = Math.floor(arr.length / 2);
    const left = arr.slice(0, middle);
    const right = arr.slice(middle);
  
    // Recursively sort the left and right halves
    return merge(mergeSort(left), mergeSort(right));
  }
  
  function merge(left, right) {
    const merged = [];
    let i = 0;
    let j = 0;
  
    // Compare elements from left and right arrays
    while (i < left.length && j < right.length) {
      if (left[i] < right[j]) {
        merged.push(left[i]);
        i++;
      } else {
        merged.push(right[j]);
        j++;
      }
    }
  
    // Add remaining elements (if any)
    merged.push(...left.slice(i));
    merged.push(...right.slice(j));
  
    return merged;
  }
  
  // Example usage:
  const numbers = [5, 3, 8, 2, 1, 4];
  const sortedNumbers = mergeSort(numbers);
  
  console.log(sortedNumbers); // Output: [1, 2, 3, 4, 5, 8]
  