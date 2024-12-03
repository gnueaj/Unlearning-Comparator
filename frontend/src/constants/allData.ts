const fileNames0 = ["0000", "a000", "3736", "dd2c", "f48d"];
const fileNames1 = ["0001", "a001", "887e", "baa5", "bba4"];
const fileNames2 = ["0002", "a002", "0711", "8823", "ea8c"];
const fileNames3 = ["0003", "a003", "3e28", "6451", "f708"];
const fileNames4 = ["0004", "a004", "1f8c", "8829", "aa35"];
const fileNames5 = ["0005", "a005", "258f", "5ae7", "efdb"];
const fileNames6 = ["0006", "a006", "7aff", "b44d", "b54c"];
const fileNames7 = ["0007", "a007", "0a80", "e57a", "fc17"];
const fileNames8 = ["0008", "a008", "0e04", "6fc7", "92a7"];
const fileNames9 = ["0009", "a009", "7f41", "455b", "b2c6"];

const fileNames = [
  fileNames0,
  fileNames1,
  fileNames2,
  fileNames3,
  fileNames4,
  fileNames5,
  fileNames6,
  fileNames7,
  fileNames8,
  fileNames9,
];

export async function loadExperimentData(forgetClass: number) {
  try {
    const dataArray = await Promise.all(
      fileNames[forgetClass].map(
        (fileName) => import(`./data/${forgetClass}/${fileName}.json`)
      )
    );

    return dataArray.reduce(
      (acc, data) => ({
        ...acc,
        [data.default.id]: data.default,
      }),
      {}
    );
  } catch (error) {
    console.error(`Error loading data for forget class ${forgetClass}:`, error);
    throw error;
  }
}
