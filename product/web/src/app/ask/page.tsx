import { redirect } from "next/navigation";

export default function AskRedirect() {
  redirect("/?chat=1");
}
