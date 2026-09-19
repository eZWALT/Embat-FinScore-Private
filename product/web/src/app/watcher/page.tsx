import { redirect } from "next/navigation";

export default function WatcherRedirect() {
  redirect("/?modo=profundo#vigilancia");
}
