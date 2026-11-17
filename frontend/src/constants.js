export const LABEL_CLASS = { 'Good Fit': 'good', 'Potential Fit': 'potential', 'No Fit': 'nofit' }

export const DEFAULT_WEIGHTS = { skills: 40, experience: 25, semantic: 20, keyword: 15 }
export const WEIGHT_KEYS = Object.keys(DEFAULT_WEIGHTS)
export const DEFAULT_TUNE = {
  mode: 'model', weights: DEFAULT_WEIGHTS, mustHave: [], minYears: '', unknownYears: 'keep', topK: 5,
}

export function isDefaultTune(t, topK) {
  return t.mode === 'model' && !t.mustHave.length && t.minYears === '' && t.topK === topK &&
    WEIGHT_KEYS.every((k) => t.weights[k] === DEFAULT_WEIGHTS[k])
}
