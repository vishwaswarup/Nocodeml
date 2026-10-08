/** Facts shown on the legal pages. One place to change them. */
export const SITE = {
  product: "NoCodeML",
  operator: "Vishwaswarup Rath",
  email: "nocodemachinelearning@gmail.com",
  updated: "8 October 2026",
} as const;

/** Where the Colab notebook lives. Opening it through this link needs the GitHub repository to be public. */
export const COLAB_NOTEBOOK = {
  openUrl: "https://colab.research.google.com/github/vishwaswarup/Nocodeml/blob/master/web/public/nocodeml-colab.ipynb",
  downloadPath: "/nocodeml-colab.ipynb",
} as const;
