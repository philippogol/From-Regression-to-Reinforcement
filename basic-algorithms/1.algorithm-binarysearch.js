function binarySearch(array, target) {
    // Set initial starting and ending indices
    let startIndex = 0;
    let endIndex = array.length - 1;
  
    // Loop until the search range collapses (start >= end)
    while (startIndex <= endIndex) {
      // Calculate the middle index
      const middleIndex = Math.floor((startIndex + endIndex) / 2);
  
      // Check if the target is at the middle element
      if (array[middleIndex] === target) {
        return middleIndex; // Target found, return its index
      } else if (array[middleIndex] < target) {
        // Target is greater, search the right half
        startIndex = middleIndex + 1;
      } else {
        // Target is less, search the left half
        endIndex = middleIndex - 1;
      }
    }
  
    // If loop exits without finding the target, return -1
    return -1; // Target not found
  }
  
  // Example usage:
  const numbers = [1, 3, 5, 7, 9];
  const target = 7;
  
  const index = binarySearch(numbers, target);
  
  if (index !== -1) {
    console.log(`Target ${target} found at index ${index}`);
  } else {
    console.log(`Target ${target} not found in the array`);
  }
  