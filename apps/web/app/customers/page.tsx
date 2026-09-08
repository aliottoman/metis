import { Suspense } from "react";

import { CustomerWorkbench } from "@/components/customer-workbench";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";

// The workbench reads the URL, so it renders on the client; until then the
// page shows its own header and the shape of the list, never a blank pane.
export default function CustomersPage() {
  return (
    <Suspense
      fallback={
        <div className="customers" aria-busy="true">
          <PageHeader
            eyebrow="Customer intelligence"
            title="Customers"
            lede="Notes become reviewed, account-scoped facts, actions and ready-to-paste updates."
          />
          <Skeleton rows={6} height={32} />
        </div>
      }
    >
      <CustomerWorkbench />
    </Suspense>
  );
}
