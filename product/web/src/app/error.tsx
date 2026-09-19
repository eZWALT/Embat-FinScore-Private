"use client";

import { AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Error({ reset }: { reset: () => void }) {
  return (
    <main className="grid min-h-screen place-items-center px-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <AlertTriangle className="mb-2 size-5 text-muted-foreground" />
          <CardTitle>No se han podido cargar los datos</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Comprueba que los datos están disponibles y vuelve a intentarlo.
          </p>
          <Button onClick={reset}>Reintentar</Button>
        </CardContent>
      </Card>
    </main>
  );
}
