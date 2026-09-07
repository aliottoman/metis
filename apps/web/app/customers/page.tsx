import { Suspense } from "react";

import { CustomerWorkbench } from "@/components/customer-workbench";

export default function CustomersPage() {
  return (
    <Suspense fallback={<div className="customers" aria-busy="true" />}>
      <CustomerWorkbench />
    </Suspense>
  );
}

