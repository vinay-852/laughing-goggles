require('dotenv').config();
const fs = require('fs');
const csv = require('csv-parser');
const { GeminiChat } = require('@langchain/google');
const { RunnableParallel } = require('langchain/runnables');
const path = require('path');
const { parse } = require('json2csv');

// 1. Read CSV and split into batches
function readCsvWithMeta(filePath) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(filePath)) {
      return reject(new Error(`Input file not found: ${filePath}`));
    }
    const rows = [];
    let headers = null;
    let instructions = null;
    let rowCount = 0;
    fs.createReadStream(filePath)
      .on('error', err => reject(new Error(`Error reading file: ${err.message}`)))
      .pipe(csv())
      .on('headers', (hdrs) => {
        headers = hdrs;
      })
      .on('data', (data) => {
        rowCount++;
        // Skip empty rows
        if (Object.values(data).every(v => v === '')) return;
        if (rowCount === 1) {
          // First row is header, already handled
          return;
        }
        if (rowCount === 2) {
          instructions = data;
          return;
        }
        rows.push(data);
      })
      .on('end', () => {
        if (!headers || !instructions || rows.length === 0) {
          return reject(new Error('CSV file missing headers, instructions, or questions.'));
        }
        resolve({ headers, instructions, rows });
      });
  });
}

// 2. Process a batch with Gemini
async function processBatch(batch, gemini, instructions) {
  // Build a batch prompt for 100 questions at a time
  // Use LangChain's structured output: expect a JSON array of answers
  const questions = batch.map(row => row);
  const prompt = `Instructions: ${JSON.stringify(instructions)}\nQuestions: ${JSON.stringify(questions)}\n\nReturn a JSON array of answers, where each answer corresponds to the question at the same index.`;
  try {
    // Use LangChain's structured output parser
    const response = await gemini.invoke(prompt, {
      outputParser: {
        parse: (text) => {
          try {
            // Find first JSON array in response
            const match = text.match(/\[.*\]/s);
            if (match) {
              return JSON.parse(match[0]);
            }
            return [];
          } catch (err) {
            return [];
          }
        }
      }
    });
    // Attach answers to rows
    const answers = Array.isArray(response) ? response : [];
    return batch.map((row, idx) => ({ ...row, Answer: answers[idx] || '' }));
  } catch (err) {
    // If Gemini fails, fill all answers with error
    return batch.map(row => ({ ...row, Answer: `Gemini API error: ${err.message}` }));
  }
}

// 3. Main function to orchestrate
async function main() {
  const inputPath = path.resolve('input.csv');
  const outputPath = path.resolve('output_filled.csv');
  const apiKey = process.env.GEMINI_API_KEY || 'YOUR_GEMINI_API_KEY';

  if (!apiKey || apiKey === 'YOUR_GEMINI_API_KEY') {
    console.error('Error: Gemini API key is missing or invalid. Set GEMINI_API_KEY in a .env file.');
    process.exit(1);
  }

  let meta;
  try {
    meta = await readCsvWithMeta(inputPath);
  } catch (err) {
    console.error('Error reading CSV:', err.message);
    process.exit(1);
  }

  const { headers, instructions, rows } = meta;
  // Remove empty rows if any
  const filteredRows = rows.filter(row => Object.values(row).some(v => v && v.trim() !== ''));
  // Split into batches
  const batchSize = 100;
  const batches = [];
  for (let i = 0; i < filteredRows.length; i += batchSize) {
    batches.push(filteredRows.slice(i, i + batchSize));
  }

  const gemini = new GeminiChat({ apiKey });

  // Process batches in groups of 4 to avoid rate limit errors
  let results = [];
  for (let i = 0; i < batches.length; i += 4) {
    const batchGroup = batches.slice(i, i + 4);
    const parallelRunnables = batchGroup.map(batch => ({
      runnable: async () => await processBatch(batch, gemini, instructions)
    }));
    try {
      const groupResults = await RunnableParallel.run(parallelRunnables);
      results = results.concat(groupResults);
    } catch (err) {
      console.error('Error during batch processing:', err.message);
      process.exit(1);
    }
  }
  // Flatten results
  const flatResults = results.flat();
  // Compose output CSV: keep original headers, add 'Answer' column
  const outputHeaders = [...headers, 'Answer'];
  try {
    const csvOut = parse(flatResults, { fields: outputHeaders });
    fs.writeFileSync(outputPath, csvOut);
    console.log(`Results written to ${outputPath}`);
  } catch (err) {
    console.error('Error writing output file:', err.message);
    process.exit(1);
  }
}

main();
