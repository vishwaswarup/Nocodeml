"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { Button, IconButton } from "./ui/button";
import { StepNumber } from "./ui/card";
import { ChoiceList, Segmented } from "./ui/choice";
import { ProgressLine } from "./ui/data";
import { QuestionField } from "./ui/field";

/** One decision at a time, in the style of Palette's onboarding form. */
export function GuidedSteps() {
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState(1);
  const [name, setName] = useState("");
  const [target, setTarget] = useState("churn");
  const [task, setTask] = useState<"classification" | "regression">("classification");
  const total = 3;
  const go = (d: number) => {
    setDir(d);
    setStep((s) => Math.min(total - 1, Math.max(0, s + d)));
  };

  const steps = [
    {
      title: "Name your project",
      body: (
        <QuestionField label="Project name" placeholder="Customer churn" value={name}
          onChange={(e) => setName(e.target.value)} hint="You can rename it later." />
      ),
    },
    {
      title: "What should the model predict?",
      body: (
        <ChoiceList label="Target column" value={target} onChange={setTarget} options={[
          { value: "churn", label: "churn", recommended: true,
            description: "2 values (0 / 1). Looks like a label: a good classification target." },
          { value: "income", label: "income", description: "Continuous numbers: would make this a regression task." },
          { value: "customer_id", label: "customer_id", description: "Unique per row: an identifier, not something to predict." },
        ]} />
      ),
    },
    {
      title: "Confirm the task",
      body: (
        <div className="flex flex-col items-center gap-4">
          <Segmented label="Task" value={task} onChange={setTask} options={[
            { value: "classification", label: "Classification" },
            { value: "regression", label: "Regression" },
          ]} />
          <p className="max-w-md text-center text-[14px] text-fg-muted">
            NoCodeML inferred <span className="text-fg">classification</span> because the target has two distinct values.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div className="mx-auto w-full max-w-[640px]">
      <ProgressLine value={(step + 1) / total} />
      <div className="relative mt-10 min-h-[300px] overflow-hidden">
        <AnimatePresence mode="popLayout" initial={false} custom={dir}>
          <motion.div
            key={step}
            custom={dir}
            variants={{
              enter: (d: number) => ({ y: d * 40, opacity: 0 }),
              center: { y: 0, opacity: 1 },
              exit: (d: number) => ({ y: d * -40, opacity: 0 }),
            }}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            <h3 className="mb-8 flex items-center justify-center gap-3 text-[26px] tracking-[-0.02em]">
              <StepNumber n={step + 1} />
              {steps[step].title}
            </h3>
            {steps[step].body}
          </motion.div>
        </AnimatePresence>
      </div>
      <div className="mt-6 flex items-center justify-between">
        <span className="font-mono text-[12px] text-fg-subtle tabular">{step + 1} / {total}</span>
        <Button size="sm" onClick={() => go(1)} disabled={step === total - 1}>OK</Button>
        <div className="flex gap-1">
          <IconButton label="Previous" active={step > 0} disabled={step === 0} onClick={() => go(-1)}>
            <ChevronUp className="size-4" />
          </IconButton>
          <IconButton label="Next" active={step < total - 1} disabled={step === total - 1} onClick={() => go(1)}>
            <ChevronDown className="size-4" />
          </IconButton>
        </div>
      </div>
    </div>
  );
}
