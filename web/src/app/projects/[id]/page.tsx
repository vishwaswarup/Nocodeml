import { RequireAuth } from "@/components/auth-provider";
import { Workspace } from "@/components/workspace/workspace";

export default async function ProjectPage(props: PageProps<"/projects/[id]">) {
  const { id } = await props.params;
  return (
    <RequireAuth>
      <Workspace projectId={id} />
    </RequireAuth>
  );
}
