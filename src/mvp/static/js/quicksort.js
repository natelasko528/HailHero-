/**
 * Implements the Quicksort algorithm to sort an array of numbers.
 *
 * @param {Array<number>} arr The array of numbers to be sorted.
 * @returns {Array<number>} A new array containing the sorted numbers.
 */
function quicksort(arr) {
  if (arr.length <= 1) {
    return arr;
  }

  const pivot = arr[Math.floor(arr.length / 2)];
  const left = [];
  const right = [];
  const equals = [];

  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < pivot) {
      left.push(arr[i]);
    } else if (arr[i] > pivot) {
      right.push(arr[i]);
    } else {
      equals.push(arr[i]);
    }
  }

  return [...quicksort(left), ...equals, ...quicksort(right)];
}